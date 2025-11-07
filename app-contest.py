# app_contest.py
# -*- coding: utf-8 -*-

import re
from io import BytesIO
import zipfile
import xml.etree.ElementTree as ET

import pandas as pd
import streamlit as st
import fitz  # PyMuPDF


# ==========================
# Utilidades de texto / PDF
# ==========================

def normalize_spaces(s: str) -> str:
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s+\n", "\n", s)
    return s.strip()

def format_cnpj(cnpj_like: str | None) -> str | None:
    d = re.sub(r"\D", "", cnpj_like or "")
    if len(d) == 14:
        return f"{d[0:2]}.{d[2:5]}.{d[5:8]}/{d[8:12]}-{d[12:14]}"
    return cnpj_like.strip() if cnpj_like else None

def read_pdf_to_text(file_bytes: bytes) -> str:
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        return "\n".join(page.get_text("text") for page in doc)

def extract_fields(text: str) -> dict:
    raw = normalize_spaces(text)

    # CNPJ
    cnpj_m = re.search(
        r"\b(?:cnpj|c\.?n\.?p\.?j\.?)[:\s]*([0-9]{2}\.?[0-9]{3}\.?[0-9]{3}/?[0-9]{4}-?[0-9]{2})",
        raw, re.IGNORECASE
    )
    cnpj = format_cnpj(cnpj_m.group(1)) if cnpj_m else None

    # Nome da empresa (antes do CNPJ)
    empresa = None
    m1 = re.search(
        r"empresa\s+[\"“]?([^\",:\n]{3,120})[\"”]?\s*,\s*(?:cnpj|c\.?n\.?p\.?j\.?)",
        raw, flags=re.IGNORECASE
    )
    if m1:
        empresa = m1.group(1).strip()
    if not empresa and cnpj_m:
        start = max(0, cnpj_m.start() - 160)
        prev = raw[start:cnpj_m.start()]
        q = re.search(r"[\"“]([^\"”]{3,120})[\"”]\s*$", prev)
        if q:
            empresa = q.group(1).strip()
    if not empresa and cnpj_m:
        start = max(0, cnpj_m.start() - 160)
        prev = raw[start:cnpj_m.start()]
        q2 = re.search(r"([A-Z0-9&., \-]{5,120})\s*$", prev)
        if q2:
            empresa = q2.group(1).strip(" ,.-")

    # Nº do parecer (prioriza Contestação)
    parecer = None
    m_cont = re.search(
        r"parecer\s+t[eé]cnico\s+da\s+contesta[cç][aã]o\s*N[º°o]?\s*([0-9]{2,7}/[0-9]{4})",
        raw, re.IGNORECASE
    )
    if m_cont:
        parecer = m_cont.group(1)
    else:
        m_gen = re.search(
            r"parecer\s+t[eé]cnico(?:\s+[a-z/]+)?\s*n[º°o]?\s*([0-9]{2,7}/[0-9]{4})",
            raw, re.IGNORECASE
        )
        if m_gen:
            parecer = m_gen.group(1)

    # Ano-base
    ano = None
    m_ab = re.search(r"ano[ -]?base[:\s-]*([12][0-9]{3})", raw, re.IGNORECASE)
    if m_ab:
        ano = m_ab.group(1)

    return {
        "nome_da_empresa": empresa,
        "CNPJ": cnpj,
        "n_parecer": parecer,
        "ano_base": ano,
    }


# =============================================
# DOCX — preserva formatação (edita só <w:t>)
# Pág.1: placeholders padrão
# Pág.2: bloco âncora: de "Empresa:" até "Sumário" ou "Programa e Atividade de PD&I"
# =============================================
W_NS = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
ET.register_namespace('w', W_NS)

def _w(tag):  # helper p/ namespace
    return f"{{{W_NS}}}{tag}"

def _t_elems_in_p(p):
    return [(t, t.text or "") for t in p.findall('.//' + _w('t'))]

def _runs_in_p(p):
    runs = []
    for r in p.findall('.//' + _w('r')):
        t_nodes = r.findall('.//' + _w('t'))
        runs.append({'r': r, 't_nodes': t_nodes})
    return runs

def _replace_substring_in_t(t_elem, rx, repl):
    text = t_elem.text or ""
    new_text, n = rx.subn(repl, text)
    if n:
        t_elem.text = new_text
    return n

def _replace_in_same_t(t_elem, patterns):
    total = 0
    for rx, repl in patterns:
        total += _replace_substring_in_t(t_elem, rx, repl)
    return total

# ---------- BLOCO MULTI-RUNS ----------

def _concat_t_runs(p):
    full = []
    spans = []
    cur = 0
    for t in p.findall('.//' + _w('t')):
        txt = t.text or ""
        start = cur
        end = start + len(txt)
        full.append(txt)
        spans.append((t, start, end))
        cur = end
    return "".join(full), spans

def _replace_span_across_runs(spans, start, end, replacement):
    if start >= end:
        return 0
    made = 0
    for (t, s, e) in spans:
        txt = t.text or ""
        if e <= start or s >= end:
            continue
        made = 1
        left_len = max(0, start - s)
        right_cut = max(0, e - end)
        left = txt[:left_len] if left_len > 0 else ""
        right = txt[e - right_cut:] if right_cut > 0 else ""
        if s <= start < e:
            t.text = left + (replacement or "") + right
        else:
            t.text = right
    return made

def _replace_token_across_runs(p, token_rx: str, replacement: str, after_label_rx: str | None = None) -> int:
    full, spans = _concat_t_runs(p)
    start_at = 0
    if after_label_rx:
        mlabel = re.search(after_label_rx, full, re.IGNORECASE)
        if not mlabel:
            return 0
        start_at = mlabel.end()
    m = re.search(token_rx, full[start_at:], re.IGNORECASE)
    if not m:
        return 0
    s = start_at + (m.start(1) if m.groups() else m.start())
    e = start_at + (m.end(1)   if m.groups() else m.end())
    return _replace_span_across_runs(spans, s, e, replacement)

def _replace_all_tokens_across_runs(p, token_rx: str, replacement: str, after_label_rx: str | None = None, max_count: int = 10) -> int:
    total = 0
    count = 0
    while count < max_count:
        changed = _replace_token_across_runs(p, token_rx, replacement, after_label_rx)
        if not changed:
            break
        total += changed
        count += 1
    return total

# ---- Inserção após rótulos ----

def _insert_text_at_index_across_runs(spans, idx: int, text: str) -> int:
    for (t, s, e) in spans:
        if s <= idx <= e:
            local = idx - s
            base = t.text or ""
            t.text = base[:local] + text + base[local:]
            return 1
    if spans:
        t, s, e = spans[-1]
        t.text = (t.text or "") + text
        return 1
    return 0

def _replace_or_insert_after_label(p, label_rx: str, value: str,
                                   ahead_until: int = 80,
                                   consume_pattern: str = r"\s*[:\-–—\u00A0]?\s*([Xx/⁄∕\- \u2010\u2011\u2013\u2014\"“”`]{0,20})") -> int:
    if not value:
        return 0
    full, spans = _concat_t_runs(p)
    mlabel = re.search(label_rx, full, re.IGNORECASE)
    if not mlabel:
        return 0
    anchor = mlabel.end()
    comma = full.find(",", anchor)
    limit = min(len(full), anchor + ahead_until, comma if comma != -1 else len(full))
    tail = full[anchor:limit]
    mcons = re.match(consume_pattern, tail, re.IGNORECASE)
    if mcons:
        start = anchor
        end = anchor + mcons.end()
        prefix_space = "" if (start > 0 and full[start-1].isspace()) else " "
        return _replace_span_across_runs(spans, start, end, prefix_space + value)
    insert_text = (" " if (anchor > 0 and not full[anchor-1].isspace()) else "") + value
    return _insert_text_at_index_across_runs(spans, anchor, insert_text)


def _find_scope_indices(body):
    paras = body.findall('.//' + _w('p'))
    start = None
    end = None
    for i, p in enumerate(paras):
        txt = "".join(t.text or "" for t in p.findall('.//' + _w('t')))
        if start is None and re.search(r"\bEmpresa\s*:", txt, re.IGNORECASE):
            start = i
            continue
        if start is not None:
            if re.search(r"\bSum[áa]rio\b", txt, re.IGNORECASE) or \
               re.search(r"Programa\s+e\s+Atividade\s+de\s+PD&I", txt, re.IGNORECASE):
                end = i
                break
    if start is None:
        return None, None
    if end is None:
        end = min(start + 60, len(paras) - 1)
    return start, end

def _replace_page1_placeholders(p, fields):
    nome    = fields.get("nome_da_empresa") or ""
    cnpj    = fields.get("CNPJ") or ""
    parecer = fields.get("n_parecer") or ""
    ano     = fields.get("ano_base") or ""

    same_t_patterns_pg1 = []
    if nome:
        same_t_patterns_pg1 += [
            (re.compile(r"`?Nome da Empresa`?", re.IGNORECASE), nome),
            (re.compile(r"`?Raz[aã]o Social`?", re.IGNORECASE), nome),
            (re.compile(r"`?Aaa`?", re.IGNORECASE), nome),
        ]
    if parecer:
        same_t_patterns_pg1.append((re.compile(r"(?<!\d)X{4}/X{4}(?!\d)", re.IGNORECASE), parecer))
    if ano:
        same_t_patterns_pg1.append((re.compile(r"20`?X{2}`?", re.IGNORECASE), ano))
    if cnpj:
        same_t_patterns_pg1.append((re.compile(r"[` ]?[Xx]{2}\.[Xx]{3}\.[Xx]{3}/[Xx]{3,4}-[Xx]{2}[` ]?"), cnpj))

    changes = 0
    for t_elem, _ in _t_elems_in_p(p):
        changes += _replace_in_same_t(t_elem, same_t_patterns_pg1)

    if ano and len(ano) == 4:
        last2 = ano[-2:]
        rx_xx_only = re.compile(r"^\s*`?X{2}`?\s*$")
        t_list = _t_elems_in_p(p)
        for i in range(1, len(t_list)):
            prev_t, prev_txt = t_list[i-1]
            curr_t, curr_txt = t_list[i]
            if prev_txt.strip().endswith("20") or prev_txt.strip().endswith("20`"):
                if rx_xx_only.match(curr_txt.strip()):
                    left_ws  = len(curr_txt) - len(curr_txt.lstrip(' '))
                    right_ws = len(curr_txt) - len(curr_txt.rstrip(' '))
                    core = curr_txt.strip()
                    left_tick  = core.startswith("`")
                    right_tick = core.endswith("`")
                    val = last2
                    if left_tick:  val = "`" + val
                    if right_tick: val = val + "`"
                    curr_t.text = (" " * left_ws) + val + (" " * right_ws)
                    changes += 1
    return changes

def _replace_cnpj_after_label_in_paragraph(p, cnpj: str) -> int:
    if not cnpj:
        return 0
    runs = _runs_in_p(p)
    tail = ""
    label_seen = False
    for r in runs:
        for t in r['t_nodes']:
            txt = t.text or ""
            combo = (tail + txt).lower()
            if "cnpj" in combo:
                label_seen = True
                break
            tail = txt[-3:].lower()
        if label_seen:
            break
    if not label_seen:
        return 0
    rx_placeholder = re.compile(r"[Xx\.\-\/` \u00A0\u202F]+")
    changes = 0
    placed = False
    saw_label = False
    tail = ""
    for r in runs:
        for t in r['t_nodes']:
            txt = t.text or ""
            combo = (tail + txt).lower()
            if not saw_label and "cnpj" in combo:
                saw_label = True
            tail = txt[-3:].lower()
            if not saw_label or not txt:
                continue
            pos = 0
            while True:
                m = rx_placeholder.search(txt, pos)
                if not m:
                    break
                seg = txt[m.start():m.end()]
                if seg.count('X') + seg.count('x') >= 2:
                    if not placed:
                        txt = txt[:m.start()] + cnpj + txt[m.end():]
                        t.text = txt
                        pos = m.start() + len(cnpj)
                        placed = True
                        changes += 1
                    else:
                        txt = txt[:m.start()] + "" + txt[m.end():]
                        t.text = txt
                        pos = m.start()
                        changes += 1
                else:
                    pos = m.end()
    return changes

def fill_docx_p1_and_block_p2(docx_bytes: bytes, fields: dict, address: dict | None = None, debug: bool = False):
    nome    = fields.get("nome_da_empresa") or ""
    cnpj    = fields.get("CNPJ") or ""
    parecer = fields.get("n_parecer") or ""
    ano     = fields.get("ano_base") or ""

    logradouro = (address or {}).get("logradouro") or ""
    numero_end = (address or {}).get("numero") or ""
    cep_end    = (address or {}).get("cep") or ""
    cidade     = (address or {}).get("cidade") or ""
    uf         = (address or {}).get("estado") or ""
    cidade_uf  = f"{cidade} – {uf}" if (cidade and uf) else ""

    rx_empresa_label = re.compile(r"\bEmpresa\s*:", re.IGNORECASE)
    rx_aaa           = re.compile(r"\bAaa\b")
    rx_razao         = re.compile(r"Raz[aã]o Social", re.IGNORECASE)

    zin = zipfile.ZipFile(BytesIO(docx_bytes), 'r')
    out_buf = BytesIO()
    zout = zipfile.ZipFile(out_buf, 'w', compression=zipfile.ZIP_DEFLATED)
    total_changes = 0
    dbg_rows = []

    for name in zin.namelist():
        data = zin.read(name)

        if name != 'word/document.xml':
            zout.writestr(name, data)
            continue

        root = ET.fromstring(data)
        body = root.find('.//' + _w('body'))
        if body is None:
            zout.writestr(name, data)
            continue

        p_list = body.findall('.//' + _w('p'))
        start_idx, end_idx = _find_scope_indices(body)

        for i, p in enumerate(p_list):
            par_text = "".join(t.text or "" for t in p.findall('.//' + _w('t')))

            if start_idx is not None and end_idx is not None and start_idx <= i <= end_idx:
                before = par_text

                # Empresa: Aaa
                if rx_empresa_label.search(par_text) and nome:
                    for t in p.findall('.//' + _w('t')):
                        _ = _replace_substring_in_t(t, rx_aaa, nome)

                # Razão Social
                if nome:
                    for t in p.findall('.//' + _w('t')):
                        _ = _replace_substring_in_t(t, rx_razao, nome)

                # CNPJ
                if cnpj:
                    total_changes += _replace_cnpj_after_label_in_paragraph(p, cnpj)
                    for t in p.findall('.//' + _w('t')):
                        _ = _replace_substring_in_t(
                            t, re.compile(r"[` ]?[Xx]{2}\.[Xx]{3}\.[Xx]{3}/[Xx]{3,4}-[Xx]{2}[` ]?"), cnpj
                        )

                # Nº do parecer — cobre `xxxx/xxxx` (minúsculas ou maiúsculas), com/sem aspas/crases, quebrado ou não
                if parecer:
                    # 1) todas as ocorrências (across runs) — aceita /, ⁄ (U+2044) ou ∕ (U+2215)
                    rx_par = r"([`\"“” ]*[xX]{3,6}\s*[/⁄∕]\s*[xX]{3,6}[`\"“” ]*)"
                    total_changes += _replace_all_tokens_across_runs(p, rx_par, parecer, after_label_rx=None, max_count=12)
                    # 2) fallback no mesmo <w:t>
                    for t in p.findall('.//' + _w('t')):
                        _ = _replace_substring_in_t(t, re.compile(r"(?i)`?[\"“”]?[x]{3,6}\s*[/⁄∕]\s*[x]{3,6}[\"“”]?`?"), parecer)
                    # 3) inserir após labels se não houver placeholder
                    total_changes += _replace_or_insert_after_label(
                        p, label_rx=r"Contesta[cç][aã]o\s*n[º°o]\s*", value=parecer,
                        consume_pattern=r"\s*([xX/⁄∕\- \"“”]{0,20})"
                    )
                    total_changes += _replace_or_insert_after_label(
                        p, label_rx=r"Parecer\s+T[eé]cnico\s*n[º°o]\s*", value=parecer,
                        consume_pattern=r"\s*([xX/⁄∕\- \"“”]{0,20})"
                    )
                    # 4) fallback genérico: qualquer 'nº' no bloco
                    total_changes += _replace_or_insert_after_label(
                        p, label_rx=r"n[º°o]\s*", value=parecer,
                        consume_pattern=r"\s*([xX/⁄∕\- \"“”]{0,20})"
                    )

                # Ano-base — 20XX (ou 20`XX`) e fallback para '20' após label
                if ano:
                    rx_20xx_any = r"(20\s*`?X{2}`?)"
                    total_changes += _replace_all_tokens_across_runs(p, rx_20xx_any, ano, after_label_rx=None, max_count=6)
                    for t in p.findall('.//' + _w('t')):
                        _ = _replace_substring_in_t(t, re.compile(r"20`?X{2}`?", re.IGNORECASE), ano)
                    total_changes += _replace_or_insert_after_label(
                        p, label_rx=r"ano-?base\s*", value=ano,
                        consume_pattern=r"\s*(?:[:\-–—]?\s*)?(20)(?!\d)"
                    )

                # Endereço — CEP, Cidade–Estado, Rua, Número
                if cep_end:
                    rx_label_cep = r"\bCEP\b"
                    rx_token_cep = r"([` ]*[Xx]{4}\s*[-–—\u2010\u2011]\s*[Xx]{3}[` ]*)"
                    total_changes += _replace_token_across_runs(p, rx_token_cep, cep_end, after_label_rx=rx_label_cep)
                    total_changes += _replace_or_insert_after_label(
                        p, label_rx=r"\bCEP\b\s*:?", value=cep_end,
                        consume_pattern=r"\s*[:\-–—]?\s*([Xx/\-\u2010\u2011\u2013\u2014 ]{0,20})"
                    )
                    for t in p.findall('.//' + _w('t')):
                        _ = _replace_substring_in_t(t, re.compile(r"`?X{4}[-–—\u2010\u2011]X{3}`?"), cep_end)

                if cidade_uf:
                    rx_token_ciduf = r"(Cidade\s*[-–—\u2010\u2011]\s*Estado)"
                    total_changes += _replace_all_tokens_across_runs(p, rx_token_ciduf, cidade_uf, None, 3)
                    for t in p.findall('.//' + _w('t')):
                        _ = _replace_substring_in_t(t, re.compile(r"`?Cidade\s*[-–—\u2010\u2011]\s*Estado`?", re.IGNORECASE), cidade_uf)

                if logradouro:
                    rx_rua = r"(Rua\/Avenida\/Estrada)"
                    total_changes += _replace_all_tokens_across_runs(p, rx_rua, logradouro, None, 2)
                    for t in p.findall('.//' + _w('t')):
                        _ = _replace_substring_in_t(t, re.compile(r"`?Rua\/Avenida\/Estrada`?", re.IGNORECASE), logradouro)

                if numero_end:
                    rx_numtoken = r"(N[úu]mero)"
                    total_changes += _replace_all_tokens_across_runs(p, rx_numtoken, numero_end, None, 2)
                    for t in p.findall('.//' + _w('t')):
                        _ = _replace_substring_in_t(t, re.compile(r"`?N[úu]mero`?", re.IGNORECASE), numero_end)

                after = "".join(t.text or "" for t in p.findall('.//' + _w('t')))
                if before != after:
                    total_changes += 1
                    if debug:
                        dbg_rows.append({"escopo": "p2-bloco", "par_idx": i, "antes": before[:2000], "depois": after[:2000]})

            else:
                if i < 80:
                    ch = _replace_page1_placeholders(p, fields)
                    if ch:
                        total_changes += ch
                        if debug:
                            after = "".join(t.text or "" for t in p.findall('.//' + _w('t')))
                            dbg_rows.append({"escopo": "p1", "par_idx": i, "antes": par_text[:2000], "depois": after[:2000]})

        zout.writestr(name, ET.tostring(root, encoding='utf-8', method='xml'))

    zout.close()
    out_buf.seek(0)
    return out_buf.getvalue(), total_changes, dbg_rows


# ==========================
# CERTIDÃO (PDF com texto)
# ==========================

def read_pdf_text_simple(file_bytes: bytes) -> str:
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        return "\n".join(page.get_text("text") for page in doc)

def extract_address_from_certificate_text(raw_text: str) -> dict:
    text = raw_text.replace("\r", "")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines() if ln.strip()]
    text = "\n".join(lines)
    text = text.replace("\u2010", "-").replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")

    flags = re.IGNORECASE | re.MULTILINE

    start = re.search(r"^\s*(LOGRADOURO|ENDERE[ÇC]O)\b.*$", text, flags)
    end = re.search(r"^\s*(ENDERE[ÇC]O ELETR[ÔO]NICO|ENTE FEDERATIVO|SITUA[ÇC][ÃA]O CADASTRAL|MOTIVO DE SITUA[ÇC][ÃA]O CADASTRAL)\b.*$", text, flags)
    if start:
        region = text[start.start(): end.start()] if (end and end.start() > start.start()) else text[start.start():]
    else:
        region = text

    rx_logra = re.compile(r"^\s*(?:LOGRADOURO|ENDERE[ÇC]O)\s*[:\-]?\s*(.+)$", flags)
    rx_num   = re.compile(r"^\s*N[ÚU]MERO\s*[:\-]?\s*(.+)$", flags)
    rx_cep   = re.compile(r"^\s*CEP\s*[:\-]?\s*([0-9.\- ]{8,12})$", flags)
    rx_cid   = re.compile(r"^\s*(?:MUNIC[ÍI]PIO|CIDADE)\s*[:\-]?\s*(.+)$", flags)
    rx_uf    = re.compile(r"^\s*(?:UF|ESTADO)\s*[:\-]?\s*([A-Za-z]{2})$", flags)

    def _norm(s: str | None) -> str | None:
        if s is None:
            return None
        s = s.replace("*", " ")
        s = re.sub(r"[ \t]+", " ", s)
        return s.strip()

    def _normalize_cep(cep_raw: str | None) -> str | None:
        if not cep_raw:
            return None
        digits = re.sub(r"\D", "", cep_raw)
        return f"{digits[:5]}-{digits[5:]}" if len(digits) == 8 else cep_raw.strip()

    logradouro = numero = cep = cidade = uf = None

    m = rx_logra.search(region);     logradouro = _norm(m.group(1)) if m else None
    m = rx_num.search(region);       numero     = _norm(m.group(1)) if m else None
    m = rx_cep.search(region);       cep        = _normalize_cep(m.group(1)) if m else None
    m = rx_cid.search(region);       cidade     = _norm(m.group(1)) if m else None
    m = rx_uf.search(region);        uf         = _norm(m.group(1).upper()) if m else None

    if (numero is None) and logradouro:
        m2 = re.search(r"(.+?),\s*([0-9A-Z\-\/]+)\b", logradouro)
        if m2:
            logradouro, numero = _norm(m2.group(1)), _norm(m2.group(2))

    if numero is None:
        m3 = re.search(r"(?:^|\b)N[ÚU]MERO(?!\s+DE\b)\s*[:\-]?\s*([^\n]+)", text, flags)
        if m3:
            numero = _norm(m3.group(1))

    return {"logradouro": logradouro, "numero": numero, "cep": cep, "cidade": cidade, "estado": uf}


def to_table(address: dict) -> pd.DataFrame:
    return pd.DataFrame([{
        "Rua (Logradouro)": address.get("logradouro") or "—",
        "Número": address.get("numero") or "—",
        "CEP": address.get("cep") or "—",
        "Cidade": address.get("cidade") or "—",
        "Estado (UF)": address.get("estado") or "—",
    }])


# ==========================
# UI - Streamlit
# ==========================

st.set_page_config(page_title="Extrair PDF (Parecer+Certidão) & Preencher DOCX", page_icon="🧾")
st.title("🧾 Extrair variáveis do Parecer + Endereço da Certidão → preencher Word (.docx)")

# Uploads
col_top1, col_top2 = st.columns(2)
with col_top1:
    up_pdf = st.file_uploader("1) PDF do Parecer", type=["pdf"])
with col_top2:
    cert_pdf = st.file_uploader("2) Certidão da RFB (PDF com texto)", type=["pdf"])
up_docx = st.file_uploader("3) Modelo Word (.docx)", type=["docx"])

fields = None
address = None

# Extrações

def _safe_read(upload, reader):
    try:
        return reader(upload.read())
    except Exception as e:
        st.error(str(e))
        return None

if up_pdf:
    pdf_text = _safe_read(up_pdf, read_pdf_to_text)
    if pdf_text:
        fields = extract_fields(pdf_text)

if cert_pdf:
    cert_text = _safe_read(cert_pdf, read_pdf_text_simple)
    if cert_text:
        address = extract_address_from_certificate_text(cert_text)

# Exibição
if fields:
    st.subheader("🔎 Variáveis do Parecer")
    st.dataframe(pd.DataFrame([fields]), use_container_width=True)
if address:
    st.subheader("📍 Endereço extraído da Certidão")
    st.dataframe(to_table(address), use_container_width=True)

# Ajustes manuais
if fields or address:
    with st.expander("✏️ Ajustar manualmente (opcional)"):
        c1, c2 = st.columns(2)
        with c1:
            if fields is None: fields = {}
            fields["nome_da_empresa"] = st.text_input("Nome da empresa", value=(fields.get("nome_da_empresa") or ""))
            fields["n_parecer"] = st.text_input("Nº do parecer (9999/9999)", value=(fields.get("n_parecer") or ""))
        with c2:
            fields["CNPJ"] = st.text_input("CNPJ", value=(fields.get("CNPJ") or ""))
            fields["ano_base"] = st.text_input("Ano-base (YYYY)", value=(fields.get("ano_base") or ""))

        st.markdown("**Endereço (Certidão)**")
        if address is None: address = {}
        col_a1, col_a2, col_a3 = st.columns([2,1,1])
        with col_a1:
            address["logradouro"] = st.text_input("Rua (Logradouro)", value=(address.get("logradouro") or ""))
        with col_a2:
            address["numero"] = st.text_input("Número", value=(address.get("numero") or ""))
        with col_a3:
            address["cep"] = st.text_input("CEP (99999-999)", value=(address.get("cep") or ""))
        col_a4, col_a5 = st.columns(2)
        with col_a4:
            address["cidade"] = st.text_input("Cidade", value=(address.get("cidade") or ""))
        with col_a5:
            address["estado"] = st.text_input("UF (Estado)", value=(address.get("estado") or ""))

st.markdown("---")

# Preenchimento do DOCX
DEBUG = st.checkbox("Habilitar debug (comparar 'Empresa:' → 'Sumário/Programa…')", value=False)

if up_docx and fields:
    try:
        new_docx, changes, dbg = fill_docx_p1_and_block_p2(up_docx.read(), fields, address=address or {}, debug=DEBUG)
        st.success(f"Documento preenchido. Alterações aplicadas: {changes}.")
        st.download_button(
            "⬇️ Baixar DOCX preenchido",
            data=new_docx,
            file_name="recurso_administrativo_preenchido.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

        st.caption("Variáveis mapeadas ➜ valores")
        mapa = {
            "Empresa / Razão Social / Aaa": fields.get("nome_da_empresa") or "—",
            "Ano-base (20XX/XX)": fields.get("ano_base") or "—",
            "Nº do Parecer (XXXX/XXXX)": fields.get("n_parecer") or "—",
            "CNPJ (XX.XXX.XXX/XXXX-XX)": fields.get("CNPJ") or "—",
            "Rua/Avenida/Estrada": (address or {}).get("logradouro") or "—",
            "Número (token)": (address or {}).get("numero") or "—",
            "CEP (XXXX-XXX)": (address or {}).get("cep") or "—",
            "Cidade – Estado": (f"{(address or {}).get('cidade','')} – {(address or {}).get('estado','')}").strip(" –") or "—",
        }
        st.table(pd.DataFrame(list(mapa.items()), columns=["Placeholder(s)", "Valor"]))

        if DEBUG and dbg:
            st.subheader("🔬 Debug — parágrafos alterados no bloco pág.2")
            st.dataframe(pd.DataFrame(dbg))

        with st.expander("ℹ️ Notas do preenchimento"):
            st.markdown(
                """
                - O “bloco pág.2” vai do primeiro parágrafo que contém **`Empresa:`** até o primeiro que contém
                  **`Sumário`** ou **`Programa e Atividade de PD&I`** (se não achar, limita a **60 parágrafos** após `Empresa:`).  
                - Dentro do bloco, trocamos **tokens de modelo** (`Aaa`, `Razão Social`, `XX.XXX.XXX/XXXX-XX` / `.../XXX-XX`,
                  `XXXX/XXXX`, `20XX`, `Rua/Avenida/Estrada`, `Número`, `XXXX-XXX`, `Cidade – Estado`),
                  inclusive quando **quebrados** entre `<w:t>`.  
                - A formatação é **preservada**: editamos **somente** o texto dos `<w:t>`.  
                """
            )

    except Exception as e:
        st.error(f"Falha ao processar o DOCX: {e}")
elif up_docx and not fields:
    st.warning("Envie o PDF do Parecer (e opcionalmente a Certidão) antes de preencher o Word.")
