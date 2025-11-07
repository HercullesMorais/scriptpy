# app.py
import re
import io
import streamlit as st
import pandas as pd
import fitz

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

# ---- Padrões de extração (iguais à versão “que funcionou”) ----
RE_CNPJ_MASK = r"\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}"
RE_CNPJ = re.compile(rf"\b{RE_CNPJ_MASK}\b")
RE_DATE = re.compile(r"\b\d{2}/\d{2}/\d{4}\b")
RE_SUPPLIER_CNPJ = re.compile(
    rf"(?:\bCNPJ\b\s+\d+\s+)?([A-Z0-9][A-Z0-9 .,&/ºª\-À-ÿ]+?)\s+({RE_CNPJ_MASK})",
    re.IGNORECASE
)

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def preclean_headers(txt: str) -> str:
    out = txt
    for pat in HEADER_PATTERNS:
        out = re.sub(pat, " ", out, flags=re.IGNORECASE)
    # normaliza múltiplos espaços após remoção
    return norm(out)

def get_page_text(doc, pno: int, mode: str) -> str:
    if mode == "blocks":
        blocks = doc[pno].get_text("blocks")
        txt = " ".join(b[4] for b in blocks if b[4].strip())
    else:
        txt = doc[pno].get_text("text")
    txt = norm(txt.replace("\n", " "))
    return preclean_headers(txt)

def looks_like_header(s: str) -> bool:
    s_low = s.lower()
    return any(tok in s_low for tok in HEADER_TOKENS)

def extract_doc_number(tail: str) -> str | None:
    mdate = RE_DATE.search(tail)
    if not mdate:
        return None
    pre = tail[:mdate.start()].strip()
    m_ests = list(re.finditer(r"Estratégico", pre, flags=re.IGNORECASE))
    if m_ests:
        pre = pre[m_ests[-1].end():].strip()
    pre = re.sub(r"^(?:N[\.\s°ºoO]*\s*Doc\.?\s*[:\-]?\s*)", "", pre, flags=re.IGNORECASE)
    pre = re.sub(r"^(?:N[\.\s°ºoO]*\s*[:\-]?\s*)", "", pre, flags=re.IGNORECASE)
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

def extract_records(full_text: str):
    items = []
    matches = list(RE_SUPPLIER_CNPJ.finditer(full_text))
    for i, m in enumerate(matches):
        fornecedor = norm(m.group(1))
        cnpj = m.group(2)
        # descarta se parecer cabeçalho
        if looks_like_header(fornecedor):
            continue
        start = m.end()
        end = matches[i+1].start() if i + 1 < len(matches) else len(full_text)
        tail = full_text[start:end]
        ndoc = extract_doc_number(tail)
        if ndoc:
            items.append({"fornecedor": fornecedor, "cnpj_fornecedor": cnpj, "numero_doc": ndoc})
    return items

def parse_pages(doc, p_ini: int, p_fim: int, mode: str):
    rows = []
    for pno in range(p_ini-1, min(p_fim, len(doc))):
        txt = get_page_text(doc, pno, mode)
        recs = extract_records(txt)
        for r in recs:
            r["pagina"] = pno + 1
            rows.append(r)
    return rows

# ---- UI ----
st.set_page_config(page_title="Extrair Fornecedor e Nº Doc. (anti-cabeçalho)", layout="centered")
st.title("Extrair Fornecedor e Nº Doc. — com remoção de cabeçalho")

up = st.file_uploader("PDF", type=["pdf"])
c1, c2, c3 = st.columns(3)
with c1: p_ini = st.number_input("Página inicial", 1, 999, 8)
with c2: p_fim = st.number_input("Página final", 1, 999, 16)
with c3: mode = st.selectbox("Modo leitura", ["text", "blocks"], index=0)

if st.button("Extrair", type="primary"):
    if not up:
        st.warning("Envie o PDF.")
        st.stop()
    pdf_bytes = up.read()
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        rows = parse_pages(doc, int(p_ini), int(p_fim), mode)

    if not rows:
        st.error("Nenhum registro. Tente o outro modo ou ajuste as páginas.")
    else:
        seen, out = set(), []
        for r in rows:
            k = (r["fornecedor"], r["cnpj_fornecedor"], r["numero_doc"], r["pagina"])
            if k not in seen:
                seen.add(k); out.append(r)
        df = pd.DataFrame(out, columns=["fornecedor","cnpj_fornecedor","numero_doc","pagina"])
        st.success(f"Registros: {len(rows)} | Únicos: {len(df)}")
        st.dataframe(df, use_container_width=True, height=420)
        buf = io.StringIO(); df.to_csv(buf, index=False)
        st.download_button("Baixar CSV", buf.getvalue().encode("utf-8"),
                           "fornecedor_numdoc.csv", "text/csv", use_container_width=True)