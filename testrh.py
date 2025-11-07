# appRH.py
# -*- coding: utf-8 -*-
import streamlit as st
import os
import re
import shutil
import pandas as pd
import fitz  # PyMuPDF
import pdfplumber
import unicodedata
from io import BytesIO

# ========================
# CONFIGURAÇÃO DA PÁGINA
# ========================
st.set_page_config(page_title="Extração — Relação de RH (2 modelos)", layout="wide")
st.title("📄 Extração — Relação de Recursos Humanos (Modelos 1 e 2)")

# ========================
# ESTRUTURA DE PASTA DE SAÍDA (por sessão)
# ========================
if "sess_dir" not in st.session_state:
    os.makedirs("PDF", exist_ok=True)
    st.session_state["sess_dir"] = os.path.join("PDF", "sess_appRH")
    os.makedirs(st.session_state["sess_dir"], exist_ok=True)
PASTA_DESTINO = st.session_state["sess_dir"]

if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0
if "caminho_pdf" not in st.session_state:
    st.session_state["caminho_pdf"] = None

# ========================
# UI
# ========================
arquivo = st.file_uploader(
    "Envie o arquivo PDF",
    type=["pdf"],
    key=f"uploader_{st.session_state['uploader_key']}"
)
pagina_inicial = st.number_input("Página inicial da seção", min_value=1, value=5, step=1)
modo_layout = st.radio(
    "Identificação do layout",
    ["Auto (recomendado)", "Modelo 1 (rotulado)", "Modelo 2 (novo)"],
    index=0, horizontal=True
)
parar_no_total_m1 = st.checkbox("🛑 (Modelo 1) Parar quando encontrar 'Total'", value=True)
mostrar_debug = st.checkbox("🔎 Mostrar debug (opcional)", value=False)

# ========================
# CONSTANTES / PADRÕES
# ========================
CPF_RE  = r"\d{3}\.\d{3}\.\d{3}\-\d{2}"
CPF_ANY = re.compile(CPF_RE)
MONEY_RE = re.compile(r"\d{1,3}(?:\.\d{3})*,\d{2}")

# Titulações canônicas e prioridade para resolver conflitos
TIT_CANON = {
    "TECNICO DE NIVEL MEDIO": "Técnico de Nível Médio",
    "TÉCNICO DE NÍVEL MÉDIO": "Técnico de Nível Médio",
    "APOIO TECNICO": "Apoio Técnico",
    "APOIO TÉCNICO": "Apoio Técnico",
    "POS-GRADUADO": "Pós-Graduado",
    "PÓS-GRADUADO": "Pós-Graduado",
    "GRADUADO": "Graduado",
    "MESTRE": "Mestre",
    "DOUTOR": "Doutor",
}
TIT_PRIORITY = {
    "Técnico de Nível Médio": 90,
    "Apoio Técnico": 80,
    "Pós-Graduado": 70,
    "Mestre": 60,
    "Doutor": 50,
    "Graduado": 40,
}

# ========================
# FUNÇÕES UTILITÁRIAS
# ========================
def norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s or "")
    return s.casefold().strip()

def to_ascii_upper(s: str) -> str:
    s_norm = unicodedata.normalize("NFKD", s or "")
    s_noacc = "".join(c for c in s_norm if not unicodedata.combining(c))
    return s_noacc.upper()

def normaliza_espacos(t: str) -> str:
    t = (t or "").replace("\xa0", " ")
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s*/\s*", "/", t)
    return t.strip()

def norm_text_keep_lines(s: str) -> str:
    s = (s or "").replace("\xa0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    return s

def valor_para_float(valor_str: str):
    v = (valor_str or "").replace("R$", "").strip()
    v = v.replace(".", "").replace(",", ".")
    try:
        return float(v)
    except Exception:
        return None

def horas_para_int(horas_str: str):
    s = (horas_str or "").strip()
    s = s.replace(".", "").replace(",", "")
    return int(s) if s.isdigit() else None

def only_digits(x: str) -> str:
    return re.sub(r"\D", "", x or "")

def valida_cpf(cpf: str) -> bool:
    nums = only_digits(cpf)
    if len(nums) != 11 or nums == nums[0]*11:
        return False
    def dv(digs):
        s = sum(int(d)*w for d, w in zip(digs, range(len(digs)+1, 1, -1)))
        r = (s * 10) % 11
        return '0' if r == 10 else str(r)
    return nums[9] == dv(nums[:9]) and nums[10] == dv(nums[:10])

def formata_cpf(cpf: str) -> str:
    nums = only_digits(cpf)
    if len(nums) == 11:
        return f"{nums[0:3]}.{nums[3:6]}.{nums[6:9]}-{nums[9:11]}"
    return cpf or ""

def extract_valor(texto: str) -> float | None:
    cand = MONEY_RE.findall(texto or "")
    return valor_para_float(cand[-1]) if cand else None

def extract_dedic(texto: str) -> str:
    t = norm(texto or "")
    if "tempo integral" in t: return "Tempo Integral"
    if "parcial"        in t: return "Parcial"
    if "exclusiv"       in t: return "Exclusivo"
    if "integral"       in t: return "Integral"
    return ""

def canon_titulacao(txt: str) -> str:
    t = to_ascii_upper(txt or "")
    for key, canon in TIT_CANON.items():
        if key in t:
            return canon
    return ""

def better_titulacao(a: str, b: str) -> str:
    """
    Resolve conflitos entre duas titulações:
      - usa prioridade TIT_PRIORITY quando canônica,
      - senão, escolhe a mais longa (mais específica),
      - se empatar, mantém 'a'.
    """
    ca, cb = canon_titulacao(a) or a, canon_titulacao(b) or b
    pa = TIT_PRIORITY.get(ca, len(ca or ""))
    pb = TIT_PRIORITY.get(cb, len(cb or ""))
    if pb > pa:
        return cb
    if pa > pb:
        return ca
    # empate por prioridade -> escolhe a mais longa
    return a if len(a or "") >= len(b or "") else b

def strip_titulacao_from_nome(nome: str, tit: str) -> str:
    if not nome: return nome
    n_up = to_ascii_upper(nome)
    for key, canon in TIT_CANON.items():
        if key in n_up:
            # remove a expressão de titulação do Nome (sem acento)
            n_up = re.sub(key, "", n_up).strip()
    # limpeza final sobre a versão sem acento; reusa n_up como base
    n_up = re.sub(r"\b(Parcial|Integral|Exclusiv\w*)\b.*", "", n_up, flags=re.IGNORECASE)
    n_up = re.sub(r"\b\d{1,4}\b.*", "", n_up)
    n_up = re.sub(r"\s{2,}", " ", n_up).strip()
    return n_up  # retorna em ASCII-UPPER; se preferir manter acentos, adaptar com mapa reverso

# ========================
# MODELO 1 (ROTULADO) — nosso código funcional (BLOCO)
# ========================
# Regex internas (BLOCO)
RX_ITEM  = re.compile(r"\bItem\s*(\d+)\b", re.IGNORECASE)
RX_CPF   = re.compile(r"\bCPF\s*([\d.\-]+)", re.IGNORECASE)
RX_NOME  = re.compile(r"Nome\s*(.*?)\s*Titula[cç][aã]o", re.IGNORECASE | re.DOTALL)
RX_TITUL = re.compile(r"Titula[cç][aã]o\s*(.*?)\s*Total\s+Horas\s*\(Anual\)", re.IGNORECASE | re.DOTALL)
RX_HORAS = re.compile(r"Total\s+Horas\s*\(Anual\)\s*(\d+)", re.IGNORECASE)
RX_TOTAL_LINE = re.compile(r"^\s*Total\b", re.IGNORECASE)

def parse_blocks(blocks: list[str]) -> pd.DataFrame:
    """Parseia blocos (cada um iniciado por 'Item <n>') em registros."""
    rows = []

    def extract_valor_bloco(texto: str) -> float | None:
        m = re.findall(r"\d{1,3}(?:\.\d{3})*,\d{2}", texto or "")
        if m:
            v = m[-1].replace(".", "").replace(",", ".")
            try: return float(v)
            except: return None
        return None

    for b in blocks:
        m_item  = RX_ITEM.search(b)
        m_cpf   = RX_CPF.search(b)
        m_nome  = RX_NOME.search(b)
        m_titul = RX_TITUL.search(b)
        m_horas = RX_HORAS.search(b)
        if not (m_item and m_cpf):
            continue
        valor = extract_valor_bloco(b)
        if valor is None:
            continue
        rows.append({
            "Item": int(m_item.group(1)),
            "CPF": only_digits(m_cpf.group(1)),
            "Nome": re.sub(r"\s+", " ", (m_nome.group(1) if m_nome else "").strip()),
            "Titulação": re.sub(r"\s+", " ", (m_titul.group(1) if m_titul else "").strip()),
            "Total de Horas": int(m_horas.group(1)) if m_horas else None,
            "Dedicação": extract_dedic(b),
            "Valor R$": f"R$ {str(valor).replace('.', ',')}",
            "Valor (num)": valor
        })

    df = pd.DataFrame(
        rows,
        columns=["Item","CPF","Nome","Titulação","Total de Horas","Dedicação","Valor R$","Valor (num)"]
    )
    return df.sort_values("Item", kind="stable").reset_index(drop=True)

def extrair_modelo1_bloco(pdf_bytes: bytes, start_page: int, parar_no_total: bool=True) -> pd.DataFrame:
    """Varre com pdfplumber, agrupa blocos iniciados por 'Item <n>' e parseia."""
    blocks, current = [], []
    total_hit = False

    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        if start_page > len(pdf.pages):
            return pd.DataFrame()
        for pidx in range(start_page - 1, len(pdf.pages)):
            page = pdf.pages[pidx]
            txt = page.extract_text(x_tolerance=1.5, y_tolerance=3.0) or ""
            if not txt:
                continue

            for raw in txt.splitlines():
                s = (raw or "").strip()
                if not s:
                    continue
                if RX_ITEM.match(s):
                    # abre novo bloco
                    if current:
                        blocks.append("\n".join(current).strip()); current = []
                    current.append(s)
                else:
                    if current:
                        if parar_no_total and RX_TOTAL_LINE.match(s):
                            blocks.append("\n".join(current).strip()); current = []
                            total_hit = True
                            break
                        current.append(s)
            if total_hit:
                break

        if current:
            blocks.append("\n".join(current).strip())

    if not blocks:
        return pd.DataFrame()

    df = parse_blocks(blocks)
    return df

# ========================
# MODELO 2 (NOVO) — rápido por linhas (PyMuPDF) + dedup/merge + lookahead + anti-spill
# ========================
HEADER_REGEX = re.compile(
    r"3\.1\.16\.2\.8\.\s*RELAÇÃO\s+DE\s+RECURSOS\s+HUMANOS",
    re.IGNORECASE
)
TOTAL_REGEX = re.compile(
    r"^\s*TOTAL(?:\s+GERAL)?\s*(?:R\$\s*[\d\.,]+)?\s*$",
    re.IGNORECASE | re.MULTILINE
)
LABEL_AHEAD = r"(?=\s*(?:Titula[cç][aã]o|Total|Dedica[cç][aã]o|Valor)\b|$)"

RE_ITEM  = re.compile(r"^\s*Item[:\s]+(\d+)\s*$", re.IGNORECASE)
RE_CPF   = re.compile(r"^\s*CPF[:\s]+((?:\d{11})|\d{3}\.\d{3}\.\d{3}\-\d{2})\s*$", re.IGNORECASE)
RE_NOME  = re.compile(rf"^\s*Nome[:\s]+(.+?){LABEL_AHEAD}", re.IGNORECASE)
RE_TITU  = re.compile(rf"^\s*Titula[cç][aã]o[:\s]+(.+?){LABEL_AHEAD}", re.IGNORECASE)
RE_HORAS = re.compile(rf"^\s*Total(?:\s+de)?\s+Horas\s*\(Anual\)[:\s]+([\d\.,]+)\s*{LABEL_AHEAD}", re.IGNORECASE)
RE_DEDIC = re.compile(rf"^\s*Dedica[cç][aã]o[:\s]+(.+?){LABEL_AHEAD}", re.IGNORECASE)
RE_VALOR = re.compile(r"^\s*Valor\s*(?:\(R\$\))?[:\s]*R\$\s*([\d\.,]+)\s*$", re.IGNORECASE)

def extrair_modelo2_rapido(pdf_path: str, start_page: int):
    """
    Lê a seção '3.1.16.2.8. RELAÇÃO DE RECURSOS HUMANOS' (texto por página),
    monta registros por campos rotulados, e DEDUPLICA por (Item, CPF) com merge.
    """
    buffer = []
    header_encontrado = False
    total_encontrado = False

    with fitz.open(pdf_path) as doc:
        total_pag = len(doc)
        p0 = max(1, min(start_page, total_pag)) - 1

        for i in range(p0, total_pag):
            page = doc.load_page(i)
            raw = page.get_text("text") or ""
            raw = norm_text_keep_lines(raw)
            if not header_encontrado:
                m = HEADER_REGEX.search(raw)
                if not m:
                    continue
                header_encontrado = True
                trecho = raw[m.end():]
            else:
                trecho = raw

            m_tot = TOTAL_REGEX.search(trecho)
            if m_tot:
                trecho = trecho[:m_tot.start()]
                buffer.append(trecho)
                total_encontrado = True
                break
            else:
                buffer.append(trecho)

    if not header_encontrado:
        return [], "Cabeçalho '3.1.16.2.8. RELAÇÃO DE RECURSOS HUMANOS' não foi encontrado a partir da página informada.", False

    aviso = None
    if not total_encontrado:
        aviso = "Aviso: 'TOTAL' não foi encontrado; extraindo até o fim do documento."

    texto_secao = "\n".join(buffer)
    linhas = [ln.strip() for ln in texto_secao.splitlines() if ln.strip()]

    registros = []
    index = {}  # (item_int, cpf_digits) -> idx em registros
    atual = None

    def merge_records(old: dict, new: dict) -> dict:
        out = old.copy()
        # Nome: mais longo
        if (not out.get("Nome")) or (len(new.get("Nome") or "") > len(out.get("Nome") or "")):
            if new.get("Nome"): out["Nome"] = new["Nome"]
        # Titulação: resolução por prioridade/canônico
        if new.get("Titulação"):
            if not out.get("Titulação"):
                out["Titulação"] = new["Titulação"]
            else:
                out["Titulação"] = better_titulacao(out["Titulação"], new["Titulação"])
        # Horas: maior não-nulo
        try:
            ha = int(out.get("Total de Horas")) if out.get("Total de Horas") not in (None, "") else None
        except Exception:
            ha = None
        try:
            hb = int(new.get("Total de Horas")) if new.get("Total de Horas") not in (None, "") else None
        except Exception:
            hb = None
        if ha is None and hb is not None:
            out["Total de Horas"] = hb
        elif ha is not None and hb is not None:
            out["Total de Horas"] = max(ha, hb)
        # Dedicação
        if not out.get("Dedicação") and new.get("Dedicação"):
            out["Dedicação"] = new["Dedicação"]
        # Valor
        va = out.get("Valor (num)")
        vb = new.get("Valor (num)")
        if va is None and vb is not None:
            out["Valor (num)"] = vb
            out["Valor R$"] = new.get("Valor R$") or out.get("Valor R$")
        elif va is not None and vb is not None and vb != va:
            if vb > va:
                out["Valor (num)"] = vb
                out["Valor R$"] = new.get("Valor R$") or out.get("Valor R$")
        return out

    def push_atual():
        nonlocal atual, registros, index
        if not atual:
            return
        item_ok = atual.get("Item"); cpf_ok = atual.get("CPF")

        # Normalizações finais
        if atual.get("Valor R$"):
            atual["Valor (num)"] = valor_para_float(atual["Valor R$"])
        else:
            atual["Valor (num)"] = None
        if cpf_ok:
            atual["CPF"] = formata_cpf(cpf_ok)
        if atual.get("Total de Horas") is not None:
            h = horas_para_int(str(atual["Total de Horas"]))
            if h is not None:
                atual["Total de Horas"] = h
        # Canoniza titulação (ex.: Pos-Graduado -> Pós-Graduado)
        if atual.get("Titulação"):
            can = canon_titulacao(atual["Titulação"])
            if can:
                atual["Titulação"] = can
        # Anti-spill: mover titulação do Nome se houver
        if atual.get("Nome"):
            emb = canon_titulacao(atual["Nome"])
            if emb:
                atual["Titulação"] = better_titulacao(atual.get("Titulação") or "", emb)
                atual["Nome"] = strip_titulacao_from_nome(atual["Nome"], atual["Titulação"])

        atual.setdefault("Titulação", "")
        atual.setdefault("Dedicação", "")

        if item_ok and (cpf_ok or atual.get("Nome")):
            try:
                item_int = int(atual["Item"])
            except Exception:
                item_int = None
            cpf_digits = re.sub(r"\D", "", atual.get("CPF") or "")
            key = (item_int, cpf_digits if cpf_digits else None)
            if key[0] is not None and key[1]:
                if key in index:
                    j = index[key]
                    registros[j] = merge_records(registros[j], atual.copy())
                else:
                    index[key] = len(registros)
                    registros.append(atual.copy())
            else:
                registros.append(atual.copy())

        atual = None

    i = 0
    n = len(linhas)
    while i < n:
        ln = linhas[i]
        nxt = linhas[i+1] if i+1 < n else ""
        comb = (ln + " " + nxt).strip()

        # Item
        m_item = RE_ITEM.match(ln)
        if m_item:
            push_atual()
            atual = {
                "Item": m_item.group(1),
                "CPF": "",
                "Nome": "",
                "Titulação": "",
                "Total de Horas": None,
                "Dedicação": "",
                "Valor R$": None,
            }
            i += 1
            continue

        if not atual:
            i += 1
            continue

        # CPF
        if not atual["CPF"]:
            m = RE_CPF.match(ln) or RE_CPF.match(comb)
            if m:
                atual["CPF"] = m.group(1)
                i += 2 if (not RE_CPF.match(ln) and nxt) else 1
                continue

        # Nome (com lookahead: não pega a próxima label)
        if not atual["Nome"]:
            m = RE_NOME.match(ln) or RE_NOME.match(comb)
            if m:
                atual["Nome"] = m.group(1).strip()
                i += 2 if (not RE_NOME.match(ln) and nxt) else 1
                continue

        # Titulação (com lookahead)
        if not atual["Titulação"]:
            m = RE_TITU.match(ln) or RE_TITU.match(comb)
            if m:
                atual["Titulação"] = (canon_titulacao(m.group(1)) or m.group(1)).strip()
                i += 2 if (not RE_TITU.match(ln) and nxt) else 1
                continue

        # Total de Horas
        if atual["Total de Horas"] is None:
            m = RE_HORAS.match(ln) or RE_HORAS.match(comb)
            if m:
                atual["Total de Horas"] = m.group(1)
                i += 2 if (not RE_HORAS.match(ln) and nxt) else 1
                continue

        # Dedicação
        if not atual["Dedicação"]:
            m = RE_DEDIC.match(ln) or RE_DEDIC.match(comb)
            if m:
                atual["Dedicação"] = m.group(1).strip()
                i += 2 if (not RE_DEDIC.match(ln) and nxt) else 1
                continue

        # Valor
        if atual["Valor R$"] is None:
            m = RE_VALOR.match(ln) or RE_VALOR.match(comb)
            if m:
                valor_str = m.group(1)
                atual["Valor R$"] = f"R$ {valor_str}"
                push_atual()
                i += 2 if (not RE_VALOR.match(ln) and nxt) else 1
                continue

        i += 1

    # último
    push_atual()

    return registros, aviso, total_encontrado

# ========================
# PÓS-PROCESSAMENTO (DF): dedup + anti-spill final
# ========================
def pos_process_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    # Ordena
    if "Item" in df.columns:
        df["Item"] = pd.to_numeric(df["Item"], errors="coerce")
        df = df.sort_values(["Item"], kind="stable")
    if "Total de Horas" in df.columns:
        df["Total de Horas"] = pd.to_numeric(df["Total de Horas"], errors="coerce")

    # Deduplicação final por (Item, CPF)
    if set(["Item", "CPF"]).issubset(df.columns):
        df["_cpf_digits"] = df["CPF"].astype(str).str.replace(r"\D", "", regex=True)

        def pick_best(group: pd.DataFrame) -> pd.Series:
            g = group.copy()
            # Canoniza titulação em todas as linhas
            g["Titulação"] = g["Titulação"].apply(lambda x: canon_titulacao(x) or (x or ""))
            # Score por Valor e Horas
            g["_val_ok"] = g["Valor (num)"].notna().astype(int) if "Valor (num)" in g.columns else 0
            g["_hor_ok"] = g["Total de Horas"].fillna(-1).astype(int) if "Total de Horas" in g.columns else 0
            g = g.sort_values(by=["_val_ok", "_hor_ok"], ascending=[False, False])
            row = g.iloc[0].copy()
            # mescla campos e resolve Titulação
            for _, r in g.iloc[1:].iterrows():
                # Nome: mais longo
                if (not row.get("Nome")) or (len(str(r.get("Nome") or "")) > len(str(row.get("Nome") or ""))):
                    if r.get("Nome"):
                        row["Nome"] = r["Nome"]
                # Titulação: resolver conflito
                if r.get("Titulação"):
                    row["Titulação"] = better_titulacao(row.get("Titulação") or "", r["Titulação"])
                # Horas: maior
                if "Total de Horas" in g.columns and pd.notna(r.get("Total de Horas")):
                    row["Total de Horas"] = max(row.get("Total de Horas") or 0, int(r["Total de Horas"]))
                # Dedicação: preenche se vazio
                if "Dedicação" in g.columns and not row.get("Dedicação") and r.get("Dedicação"):
                    row["Dedicação"] = r["Dedicação"]
                # Valor: preferir numérico maior
                if "Valor (num)" in g.columns and pd.notna(r.get("Valor (num)")):
                    if pd.isna(row.get("Valor (num)")) or r["Valor (num)"] > row["Valor (num)"]:
                        row["Valor (num)"] = r["Valor (num)"]
                        row["Valor R$"] = r.get("Valor R$") or row.get("Valor R$")
            return row.drop(labels=[c for c in ["_val_ok","_hor_ok"] if c in row.index])

        df = (
            df.groupby(["Item","_cpf_digits"], as_index=False, group_keys=False)
                .apply(pick_best)
              .drop(columns=["_cpf_digits"], errors="ignore")
              .reset_index(drop=True)
              .sort_values(["Item"], kind="stable")
        )

    # Anti-spill final: mover titulação do Nome (se existir)
    def anti_spill_apply(row):
        nome = row.get("Nome") or ""
        tit = row.get("Titulação") or ""
        emb = canon_titulacao(nome)
        if emb:
            row["Titulação"] = better_titulacao(tit, emb)
            row["Nome"] = strip_titulacao_from_nome(nome, row["Titulação"])
        # normaliza titulação canônica
        if row.get("Titulação"):
            can = canon_titulacao(row["Titulação"])
            if can: row["Titulação"] = can
        return row

    df = df.apply(anti_spill_apply, axis=1)

    return df

# ========================
# RESET
# ========================
def reset_app():
    try:
        sess_dir = st.session_state.get("sess_dir")
        if sess_dir and os.path.exists(sess_dir):
            shutil.rmtree(sess_dir, ignore_errors=True)
    except Exception:
        pass
    st.session_state.clear()
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()

# ========================
# EXECUÇÃO
# ========================
if arquivo is not None:
    # Salva o PDF
    safe_name = os.path.basename(arquivo.name)
    caminho_pdf = os.path.join(PASTA_DESTINO, safe_name)
    with open(caminho_pdf, "wb") as f:
        f.write(arquivo.getbuffer())
    st.session_state["caminho_pdf"] = caminho_pdf
    st.success(f"📁 Arquivo salvo em: {caminho_pdf}")

    if st.button("🔍 Extrair dados"):
        pdf_bytes = open(caminho_pdf, "rb").read()

        df_final = None
        layout_escolhido = None
        aviso_m2 = None

        if modo_layout == "Modelo 1 (rotulado)":
            df_m1 = extrair_modelo1_bloco(pdf_bytes, start_page=int(pagina_inicial), parar_no_total=parar_no_total_m1)
            if not df_m1.empty:
                df_final = df_m1
                layout_escolhido = "Modelo 1 (rotulado)"

        elif modo_layout == "Modelo 2 (novo)":
            registros_m2, aviso_m2, _ = extrair_modelo2_rapido(caminho_pdf, start_page=int(pagina_inicial))
            if registros_m2:
                df_final = pd.DataFrame(registros_m2)
                layout_escolhido = "Modelo 2 (novo)"

        else:  # Auto (recomendado)
            # 1) Modelo 1
            df_m1 = extrair_modelo1_bloco(pdf_bytes, start_page=int(pagina_inicial), parar_no_total=parar_no_total_m1)
            # 2) Modelo 2
            registros_m2, aviso_m2, _ = extrair_modelo2_rapido(caminho_pdf, start_page=int(pagina_inicial))
            df_m2 = pd.DataFrame(registros_m2) if registros_m2 else pd.DataFrame()

            # Escolha por "score" (CPF válido + tem valor)
            def score_df(df: pd.DataFrame) -> int:
                if df is None or df.empty:
                    return 0
                pts = 0
                for _, r in df.iterrows():
                    has_val = (pd.notna(r.get("Valor (num)")) and r.get("Valor (num)") is not None) or bool(r.get("Valor R$"))
                    if valida_cpf(r.get("CPF", "")) and has_val:
                        pts += 1
                return pts

            s1 = score_df(df_m1)
            s2 = score_df(df_m2)

            if s2 > s1 or (s2 == s1 and len(df_m2) > len(df_m1)):
                df_final = df_m2 if not df_m2.empty else df_m1
                layout_escolhido = "Modelo 2 (novo)" if not df_m2.empty else "Modelo 1 (rotulado)"
            else:
                df_final = df_m1 if not df_m1.empty else df_m2
                layout_escolhido = "Modelo 1 (rotulado)" if not df_m1.empty else "Modelo 2 (novo)"

        # Pós-processamento / Export
        if df_final is not None and not df_final.empty:
            df_final = pos_process_df(df_final)

            if aviso_m2:
                st.info(aviso_m2)

            st.success(f"✅ Extraídos {len(df_final)} registro(s). Layout escolhido: **{layout_escolhido or '—'}**.")
            st.dataframe(df_final, use_container_width=True)

            caminho_excel = os.path.join(PASTA_DESTINO, "tabela_reconstruida.xlsx")
            caminho_csv   = os.path.join(PASTA_DESTINO, "tabela_reconstruida.csv")
            try:
                df_final.to_excel(caminho_excel, index=False)
            except Exception:
                st.warning("⚠️ Excel indisponível (openpyxl ausente). Baixe o CSV.")
            df_final.to_csv(caminho_csv, index=False, encoding="utf-8-sig")

            col1, col2, col3 = st.columns(3)
            with col1:
                if os.path.exists(caminho_excel):
                    with open(caminho_excel, "rb") as f:
                        st.download_button("📥 Baixar Excel", data=f.read(), file_name="tabela_reconstruida.xlsx", use_container_width=True)
                else:
                    st.button("📄 Excel indisponível", disabled=True, use_container_width=True)
            with col2:
                with open(caminho_csv, "rb") as f:
                    st.download_button("📥 Baixar CSV", data=f.read(), file_name="tabela_reconstruida.csv", use_container_width=True)
            with col3:
                if st.button("🔄 Nova extração", use_container_width=True):
                    reset_app()

            if mostrar_debug:
                st.subheader("🔎 Debug")
                st.write({"Layout": layout_escolhido, "Registros": len(df_final)})
                st.code(df_final.head(15).to_string(index=False))
        else:
            st.error("Nenhum registro foi encontrado. Verifique a página inicial e o layout selecionado.")
            if st.button("🔄 Nova extração"):
                reset_app()
else:
    st.info("Envie um PDF e clique em **Extrair dados** para iniciar.")