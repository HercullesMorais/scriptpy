# extrator_nf.py
# Módulo de extração para "Serviços Terceiros" (páginas 8–16 ou conforme intervalo).
# Expõe: parse_pdf(pdf_bytes: bytes, p_ini: int, p_fim: int, mode: str) -> list[dict]
# Retorna registros com: fornecedor, cnpj_fornecedor, numero_doc, pagina

import re
from typing import List, Dict

import fitz  # PyMuPDF

# ---- Cabeçalhos conhecidos (remoção preventiva) ----
HEADER_PATTERNS = [
    r"Centro\s+de\s+Custo\s+N[º°]\s+Conta\s+Cont[áa]bil\s+N[.\sº°]*Projeto\s+N[.\sº°]*Doc\.?\s+Data\s+Doc\.?\s+Disp[eê]ndio",
    r"Prestador\s+de\s+servi[cç]os\s+CNPJ\s+Descri[cç][aã]o\s+dos\s+Trabalhos\s+N[º°]\s+Centro\s+de\s+Custo\s+N[.\sº°]*Doc\.?\s+Data\s+Doc\.?\s+Disp[eê]ndio",
    r"N[.\sº°]*Projeto\s+Estrat[ée]gico/Não\s+Estrat[ée]gico\s+Tipologia\s+de\s+Disp[eê]ndio\s+Prestador\s+de\s+servi[cç]os\s+CNPJ\s+Descri[cç][aã]o\s+dos\s+Trabalhos\s+N[º°]\s+Centro\s+de\s+Custo\s+N[.\sº°]*Doc\.?\s+Data\s+Doc\.?\s+Disp[eê]ndio",
]

# Palavras que denunciam cabeçalho no campo fornecedor
HEADER_TOKENS = {
    "centro de custo", "conta contábil", "projeto", "nº doc", "n.º projeto",
    "data doc", "dispêndio", "prestador de serviços", "descrição dos trabalhos"
}

# ---- Padrões de extração ----
RE_CNPJ_MASK = r"\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}"
RE_CNPJ = re.compile(rf"\b{RE_CNPJ_MASK}\b")
RE_DATE = re.compile(r"\b\d{2}/\d{2}/\d{4}\b")
RE_SUPPLIER_CNPJ = re.compile(
    rf"(?:\bCNPJ\b\s+\d+\s+)?([A-Z0-9][A-Z0-9 .,&/ºª\-À-ÿ]+?)\s+({RE_CNPJ_MASK})",
    re.IGNORECASE
)

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def _preclean_headers(txt: str) -> str:
    out = txt
    for pat in HEADER_PATTERNS:
        out = re.sub(pat, " ", out, flags=re.IGNORECASE)
    return _norm(out)

def _get_page_text(doc, pno: int, mode: str) -> str:
    if mode == "blocks":
        blocks = doc[pno].get_text("blocks")
        txt = " ".join(b[4] for b in blocks if b[4].strip())
    else:
        txt = doc[pno].get_text("text")
    txt = _norm(txt.replace("\n", " "))
    return _preclean_headers(txt)

def _looks_like_header(s: str) -> bool:
    s_low = (s or "").lower()
    return any(tok in s_low for tok in HEADER_TOKENS)

def _extract_doc_number(tail: str) -> str | None:
    # 1) até a primeira data (Data Doc.)
    mdate = RE_DATE.search(tail)
    if not mdate:
        return None
    pre = tail[:mdate.start()].strip()

    # 2) após a última ocorrência de 'Estratégico'
    m_ests = list(re.finditer(r"Estratégico", pre, flags=re.IGNORECASE))
    if m_ests:
        pre = pre[m_ests[-1].end():].strip()

    # 3) limpar prefixos 'Nº Doc' / 'Nº' (sem impor formato)
    pre = re.sub(r"^(?:N[\.\s°ºoO]*\s*Doc\.?\s*[:\-]?\s*)", "", pre, flags=re.IGNORECASE)
    pre = re.sub(r"^(?:N[\.\s°ºoO]*\s*[:\-]?\s*)", "", pre, flags=re.IGNORECASE)

    # 4) último token útil antes da data
    tokens = re.findall(r"[A-Za-z0-9./\-]+", pre)
    if not tokens:
        return None
    for tok in reversed(tokens):
        if RE_CNPJ.fullmatch(tok):
            continue
        if tok.upper() in {"OUTROS", "APOIO", "TECNICO", "TÉCNICO", "SERVIÇO", "SERVIÇOS"}:
            continue
        return tok
    return None

def _extract_records(full_text: str) -> List[Dict]:
    items: List[Dict] = []
    matches = list(RE_SUPPLIER_CNPJ.finditer(full_text))
    for i, m in enumerate(matches):
        fornecedor = _norm(m.group(1))
        cnpj = m.group(2)
        if _looks_like_header(fornecedor):
            continue
        start = m.end()
        end = matches[i+1].start() if i + 1 < len(matches) else len(full_text)
        tail = full_text[start:end]
        ndoc = _extract_doc_number(tail)
        if ndoc:
            items.append({"fornecedor": fornecedor, "cnpj_fornecedor": cnpj, "numero_doc": ndoc})
    return items

def parse_pdf(pdf_bytes: bytes, p_ini: int, p_fim: int, mode: str = "text") -> List[Dict]:
    """
    Função pública esperada pelo app.py
    """
    rows: List[Dict] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        p_ini = max(1, int(p_ini))
        p_fim = min(len(doc), int(p_fim))
        for pno in range(p_ini - 1, p_fim):
            txt = _get_page_text(doc, pno, mode)
            recs = _extract_records(txt)
            for r in recs:
                r["pagina"] = pno + 1
                rows.append(r)
    return rows