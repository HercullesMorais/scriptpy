# modulos/extrator_form.py
from __future__ import annotations
import re
from typing import Dict, Optional, List, Tuple
import fitz  # PyMuPDF

# ============================================================
# ÂNCORAS 3.1.x
# ============================================================
def _anchor(n: str) -> re.Pattern:
    a, b, c = n.split(".")
    return re.compile(fr"{a}\.\s*{b}\.\s*{c}\.\s*", re.IGNORECASE)

ANCHORS = {
    k: _anchor(k) for k in (
        "3.1.1", "3.1.2", "3.1.3", "3.1.4", "3.1.7", "3.1.8", "3.1.9",
        "3.1.10", "3.1.11", "3.1.12", "3.1.13", "3.1.14"
    )
}

# ============================================================
# REGEX / NORMALIZAÇÃO
# ============================================================
INLINE_NEXT = re.compile(r"\s+3\.\s*1\.\s*\d{1,2}\.\s*")
PAT_NUM     = re.compile(r"\b\d+(?:\.\d+){1,}\.?\b")
LEADING_NUM = re.compile(r"^\s*\(?\s*\d+(?:\.\d+){1,}\.?\)?\s*[-–:]*\s*")

# caracteres invisíveis e variantes de colchetes/hífens
ZW_RX   = re.compile(r"[\u200B-\u200F\u202A-\u202E\u2060\uFEFF]")
WSP_RX  = re.compile(r"[ \t\u00A0]+")
DASH_RX = re.compile(r"[‐‑‒–—―]+")  # vários tipos de hífen
# [Item N] / Item N (com tolerância a colchetes fullwidth, NBSP, hífens, pontuação)
ITEM_PREFIX_RX = re.compile(
    r"^\s*(?:[\[\(【]?\s*Item\s*[\-–:]*\s*\d+\s*[\]\)】]?)\s*[:\-–]?\s*",
    re.IGNORECASE,
)
ITEM_ONLY_RX = re.compile(
    r"^\s*(?:[\[\(【]?\s*Item\s*\d+\s*[\]\)】]?)\s*$",
    re.IGNORECASE,
)

def _debracket(s: str) -> str:
    # normaliza colchetes/hífens/espacos e remove ZW chars
    s = ZW_RX.sub("", s)
    s = s.replace("【", "[").replace("】", "]")
    s = DASH_RX.sub("-", s)
    return s

# Rótulos (com tolerância a variações)
LABEL_PATS = {
    "nome_atividade": re.compile(
        r"^\s*(?:Nome\s+da\s+atividade\s+de\s+PD.?I)\s*[:\-–]*\s*",
        re.IGNORECASE),
    "pb_pa_de": re.compile(
        r"^\s*PB,\s*PA\s*ou\s*DE\s*[:\-–]*\s*",
        re.IGNORECASE),
    "area_do_projeto": re.compile(
        r"^\s*Á?A?R?E?A?\s*do\s*Projeto\s*[:\-–]*\s*|^\s*AREA\s*DO\s*PROJETO\s*[:\-–]*\s*",
        re.IGNORECASE),
    "natureza": re.compile(
        r"^\s*NATUREZA\s*[:\-–]*\s*|^\s*Natureza\s*[:\-–]*\s*",
        re.IGNORECASE),
    "destaque_elemento_novo": re.compile(
        r"^\s*Destaque\s*o?\s*elemento\s*tecnologicamente\s*novo\s*ou\s*inovador\s*da\s*atividade\s*[:\-–]*\s*",
        re.IGNORECASE),
    "barreira_desafio_tecnologico": re.compile(
        r"^\s*(?:Barreiras?\s*/?\s*Desafios?|Barreira\s*/\s*Desafio)\s*tecnol[oó]gic[oa]s?\s*[:\-–]*\s*",
        re.IGNORECASE),
    "metodologia_metodos": re.compile(
        r"^\s*Metodologia\s*/?\s*M[ée]todos?\s*[:\-–]*\s*",
        re.IGNORECASE),
    "atividade_continua_q": re.compile(
        r"^\s*Atividade\s*é\s*cont[ií]nua\??\s*[:\-–]*\s*",
        re.IGNORECASE),
    "data_inicio_atividade": re.compile(
        r"^\s*Data\s*de\s*in[ií]cio\s*da\s*atividade\s*[:\-–]*\s*",
        re.IGNORECASE),
    "previsao_termino": re.compile(
        r"^\s*Previs[aã]o\s*de\s*t[ée]rmino\s*[:\-–]*\s*",
        re.IGNORECASE),
}

# prompts alternativos
PROMPT_PATS = {
    "barreira_desafio_tecnologico": re.compile(
        r"^\s*Qual\s+a\s+barreira\s+ou\s+desafio\s+(?:tecn\w+|tenol\w+)\s+super\w+\s*[:\-–]*\s*",
        re.IGNORECASE),
    "metodologia_metodos": re.compile(
        r"^\s*(?:Qual\s*a\s*metodologia|Descreva\s*a?\s*metodologia)\s*(?:\s*/\s*m[ée]todos?)?\s*[:\-–]*\s*",
        re.IGNORECASE),
}

LEFTOVER_PATS = {
    "metodologia_metodos": re.compile(
        r"^\s*(?:utilizad[oa]s?|empregad[oa]s?|adotad[oa]s?|aplicad[oa]s?)\b\s*[:\-–,]*\s*",
        re.IGNORECASE),
}

TERMINAL_FIX_KEYS = {"destaque_elemento_novo","barreira_desafio_tecnologico","metodologia_metodos"}

def _ensure_final_full_stop(s: str) -> str:
    if not s:
        return s
    s = s.rstrip()
    if re.search(r'[.!?…][\"\'\)\]]*\s*$', s):
        return s
    return s + "."

# ============================================================
# PÁGINAS
# ============================================================
def _is_footer_line(ln: str) -> bool:
    low = ln.strip().lower()
    return (low.startswith("gerado em:")
            or "código de autenticidade" in low
            or "codigo de autenticidade" in low
            or low.startswith("página:")
            or low.startswith("pagina:"))

def _normalize_page_text(t: str) -> str:
    t = (t or "").replace("\u00A0", " ").replace("**", "")
    t = "\n".join(ln.rstrip() for ln in t.splitlines() if not _is_footer_line(ln))
    t = ZW_RX.sub("", t)
    t = re.sub(r"[ \t\r\f]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def _get_pages(file_bytes: bytes) -> List[str]:
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        return [_normalize_page_text(p.get_text("text")) for p in doc]

# ============================================================
# HELPERS DE LIMPEZA DE LINHA
# ============================================================
def _strip_prefix_and_label(s: str, label_pat: re.Pattern | None = None) -> str:
    s = _debracket(s)
    s = ITEM_PREFIX_RX.sub("", s, count=1)
    if label_pat:
        s = label_pat.sub("", s, count=1)
    return s.strip(" :-–")

def _first_useful_line(chunk: str, label_pat: re.Pattern | None = None) -> Optional[str]:
    for raw in chunk.splitlines():
        ln = raw.strip()
        if not ln:
            continue
        ln = _strip_prefix_and_label(ln, label_pat)
        if not ln or ITEM_ONLY_RX.match(ln):
            continue
        # remove numeração/resquícios tipo “3.1.x …”
        ln = INLINE_NEXT.split(ln, 1)[0].strip()
        ln = re.sub(r"\s{2,}", " ", ln).strip()
        if ln:
            return ln
    # fallback: tudo em uma linha
    all_one = " ".join(chunk.splitlines()).strip()
    all_one = _strip_prefix_and_label(all_one, label_pat)
    all_one = INLINE_NEXT.split(all_one, 1)[0].strip()
    all_one = re.sub(r"\s{2,}", " ", all_one).strip()
    return None if ITEM_ONLY_RX.match(all_one or "") else (all_one or None)

# ============================================================
# PIPELINE MULTILINHA (3.1.8/9/10, datas)
# ============================================================
def _pipeline_multiline(key: str, raw: Optional[str]) -> Optional[str]:
    if not raw:
        return raw
    s = raw
    s = _debracket(s)
    s = LEADING_NUM.sub("", s.lstrip(), 1)
    s = re.sub(r"^[\s\u2022\-\u2013\u2014/]*", "", s)
    s = ITEM_PREFIX_RX.sub("", s, 1)

    # remove rótulos / prompts
    if key in LABEL_PATS:
        s = LABEL_PATS[key].sub("", s, 1)
    if key in PROMPT_PATS:
        s = PROMPT_PATS[key].sub("", s, 1)
    if key in LEFTOVER_PATS:
        s = LEFTOVER_PATS[key].sub("", s, 1)

    # limpeza final
    s = s.lstrip(" :-–").strip()
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s or None

# ============================================================
# “Atividade é contínua?” normalização
# ============================================================
ATIVIDADE_PROMPT = re.compile(
    r"^\s*Atividade\s*é\s*cont[ií]nua\?\s*[:\-–]*\s*(?:A\s*atividade\s*é\s*cont[ií]nua\s*\(.*?\)\s*\??\s*)?",
    re.IGNORECASE | re.DOTALL
)
YES_PAT = re.compile(r"\bsim\b", re.IGNORECASE)
NO_PAT  = re.compile(r"\bn[ãa]o\b",  re.IGNORECASE)

def _normalize_atividade_continua(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return raw
    s = LEADING_NUM.sub("", raw.strip(), 1)
    s = ATIVIDADE_PROMPT.sub("", s, 1).strip()
    ms, mn = YES_PAT.search(s), NO_PAT.search(s)
    if ms and (not mn or ms.start() < mn.start()): return "Sim"
    if mn and (not ms or mn.start() < ms.start()): return "Não"
    if s.lower().startswith("sim"): return "Sim"
    if s.lower().startswith(("nao","não")): return "Não"
    return s or None

# ============================================================
# CAPTURA ENTRE ÂNCORAS
# ============================================================
def _capture_between(pages: List[str], start_pat: re.Pattern, end_pat: re.Pattern) -> Optional[str]:
    buf, cap = [], False
    for page in pages:
        if not cap:
            m = start_pat.search(page)
            if not m: continue
            s = page[m.end():]
            me = end_pat.search(s)
            if me:
                return s[:me.start()].strip() or None
            buf.append(s.strip()); cap = True
        else:
            me = end_pat.search(page)
            if me:
                prior = page[:me.start()].strip()
                if prior: buf.append(prior)
                content = "\n".join(x for x in buf if x).strip()
                return re.sub(r"\n{3,}", "\n\n", content).strip() or None
            if page: buf.append(page.strip())
    return None

# ============================================================
# PARSERS ESPECÍFICOS (single-line)
# ============================================================
def _parse_nome_atividade(block: Optional[str]) -> Optional[str]:
    if not block: return None
    # 1) primeira linha “útil” após remover [Item] e o rótulo "Nome da atividade..."
    val = _first_useful_line(block, LABEL_PATS["nome_atividade"])
    return val or None

def _parse_pb_pa_de(block: Optional[str]) -> Optional[str]:
    """
    Extrai 'PB|PA|DE' e, se houver, a descrição após o hífen:
    Ex.: 'DE - Desenvolvimento Experimental'.
    Mantém o fallback para somente 'DE' quando não houver descrição.
    """
    if not block:
        return None

    # Primeira linha útil sem [Item N] e sem o rótulo "PB, PA ou DE"
    s = _first_useful_line(block, LABEL_PATS["pb_pa_de"])
    if not s:
        return None

    s = _debracket(s)  # normaliza colchetes/hífens/espacos invisíveis

    # Captura o código (PB|PA|DE) e, opcionalmente, a descrição até o fim da linha
    # Aceita hífen normal ou tipográfico (–, —), com espaços opcionais.
    m = re.search(r"\b(PB|PA|DE)\b(?:\s*[-–—]\s*([^\n\r]+))?", s, re.IGNORECASE)
    if not m:
        # Fallback: se não casou, devolve a linha saneada
        return s.strip()

    code = m.group(1).upper()
    desc = (m.group(2) or "").strip()

    # Se houver descrição, retorna "CODE - Descrição"; senão, só "CODE"
    return f"{code} - {desc}" if desc else code

def _parse_area(block: Optional[str]) -> Optional[str]:
    if not block: return None
    s = _first_useful_line(block, LABEL_PATS["area_do_projeto"])
    return s or None

def _parse_natureza(block: Optional[str]) -> Optional[str]:
    if not block: return None
    s = _first_useful_line(block, LABEL_PATS["natureza"])
    # normaliza opções mais comuns
    if re.search(r"\bproduto\b", s, re.IGNORECASE):   return "Produto"
    if re.search(r"\bprocesso\b", s, re.IGNORECASE):  return "Processo"
    if re.search(r"\bservi[cç]o\b", s, re.IGNORECASE):return "Serviço"
    return s

# ============================================================
# MULTI-ITENS (3.1.1 repetido)
# ============================================================
def count_items(file_bytes: bytes) -> int:
    pages = _get_pages(file_bytes)
    big = "\n".join(pages)
    return len(list(ANCHORS["3.1.1"].finditer(big)))

def _slice_item_subtext(pages: List[str], item_index: int) -> Optional[str]:
    SEP = "\n<<<PAGE_BREAK>>>\n"
    big = SEP.join(pages)
    occ = list(ANCHORS["3.1.1"].finditer(big))
    if not occ or item_index < 1 or item_index > len(occ): return None
    start = occ[item_index - 1].start()
    end   = occ[item_index].start() if item_index < len(occ) else len(big)
    return big[start:end]

# ============================================================
# EXTRAÇÃO PRINCIPAL
# ============================================================
def extract_form_fields(file_bytes: bytes, item_index: Optional[int] = None) -> Dict[str, Optional[str]]:
    """
    Extrai campos do Form PD (3.1.1 a 3.1.14).
    Se item_index (1-based) for informado, extrai daquele Item apenas.
    """
    pages = _get_pages(file_bytes)

    if item_index is not None:
        sub = _slice_item_subtext(pages, item_index)
        if sub is None:
            return {k: None for k in [
                "nome_projeto","numero_projeto","pb_pa_ou_de","area_do_projeto","natureza",
                "destaque_elemento_novo","barreira_desafio_tecnologico","metodologia_metodos",
                "atividade_continua","data_inicio_atividade","previsao_termino"
            ]}
        pages = [sub]

    # blocos brutos entre âncoras
    b_nome = _capture_between(pages, ANCHORS["3.1.1"], ANCHORS["3.1.2"])
    b_num  = _capture_between(pages, ANCHORS["3.1.2"], ANCHORS["3.1.3"])
    b_pbde = _capture_between(pages, ANCHORS["3.1.3"], ANCHORS["3.1.4"])
    b_area = _capture_between(pages, ANCHORS["3.1.4"], ANCHORS["3.1.7"])
    b_nat  = _capture_between(pages, ANCHORS["3.1.7"], ANCHORS["3.1.8"])
    b_dest = _capture_between(pages, ANCHORS["3.1.8"], ANCHORS["3.1.9"])
    b_barr = _capture_between(pages, ANCHORS["3.1.9"], ANCHORS["3.1.10"])
    b_met  = _capture_between(pages, ANCHORS["3.1.10"], ANCHORS["3.1.11"])
    b_cont = _capture_between(pages, ANCHORS["3.1.11"], ANCHORS["3.1.12"])
    b_ini  = _capture_between(pages, ANCHORS["3.1.12"], ANCHORS["3.1.13"])
    b_fim  = _capture_between(pages, ANCHORS["3.1.13"], ANCHORS["3.1.14"])

    out: Dict[str, Optional[str]] = {
        "nome_projeto": _parse_nome_atividade(b_nome),
        "numero_projeto": _first_useful_line(b_num, None),  # app força "1" depois (mantido)
        "pb_pa_ou_de": _parse_pb_pa_de(b_pbde),
        "area_do_projeto": _parse_area(b_area),
        "natureza": _parse_natureza(b_nat),
        "destaque_elemento_novo": _pipeline_multiline("destaque_elemento_novo", b_dest),
        "barreira_desafio_tecnologico": _pipeline_multiline("barreira_desafio_tecnologico", b_barr),
        "metodologia_metodos": _pipeline_multiline("metodologia_metodos", b_met),
        "atividade_continua": _normalize_atividade_continua(b_cont),
        "data_inicio_atividade": _pipeline_multiline("data_inicio_atividade", b_ini),
        "previsao_termino": _pipeline_multiline("previsao_termino", b_fim),
    }

    # Pós-limpeza geral (apenas campos de texto; não mexe em nome/numero)
    for k, v in list(out.items()):
        if not isinstance(v, str) or k in {"nome_projeto","numero_projeto"}:
            continue
        v = PAT_NUM.sub("", v)
        v = v.strip(" :.-–")
        v = re.sub(r"\s{2,}", " ", v)
        v = re.sub(r"\n{3,}", "\n\n", v).strip()
        if k in TERMINAL_FIX_KEYS and v:
            v = _ensure_final_full_stop(v)
        out[k] = v or None

    return out

# ============================================================
# VISUALIZAÇÃO
# ============================================================
def as_markdown(fields: Dict[str, Optional[str]]) -> str:
    order = [
        ("nome_projeto","Nome do Projeto / Atividade (3.1.1)"),
        ("numero_projeto","Número do Projeto (3.1.2)"),
        ("pb_pa_ou_de","PB/PA/DE"),
        ("area_do_projeto","Área do Projeto"),
        ("natureza","Natureza"),
        ("destaque_elemento_novo","Destaque (elemento novo/inovador)"),
        ("barreira_desafio_tecnologico","Barreira/Desafio tecnológico"),
        ("metodologia_metodos","Metodologia/Métodos"),
        ("atividade_continua","Atividade é contínua?"),
        ("data_inicio_atividade","Data de início da atividade"),
        ("previsao_termino","Previsão de término"),
    ]
    return "\n".join(f"**{lbl}:**\n\n{fields.get(key) or ''}\n" for key,lbl in order)

# ============================================================
# DIAGNÓSTICO
# ============================================================
def scan_anchors(file_bytes: bytes) -> Dict[str, List[int]]:
    pages = _get_pages(file_bytes)
    found = {k: [] for k in ANCHORS}
    for i, txt in enumerate(pages):
        for key, pat in ANCHORS.items():
            if pat.search(txt): found[key].append(i)
    return found
__all__ = ["extract_form_fields", "as_markdown", "scan_anchors", "count_items"]