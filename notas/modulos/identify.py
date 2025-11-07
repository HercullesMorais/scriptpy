# identify.py
# -----------------------------------------------------------------------------
# Extração de "Número da Nota" (NF/NFS-e) a partir de PDFs com fallback via OCR
# + Inclusão da extração de CNPJ (com validação) usando a mesma lógica de âncoras.
# -----------------------------------------------------------------------------

import re, io, zipfile, unicodedata
from typing import List, Tuple, Optional, Dict
import fitz
from PIL import Image, ImageOps, ImageFilter
import pytesseract
import pandas as pd

# -----------------------------------------------------------------------------
# Normalização e utilitários
# -----------------------------------------------------------------------------

ZW = ["\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"]
NBSPS = ["\u00A0", "\u2007", "\u202F"]
LIG_MAP = str.maketrans({
    "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
    "“": '"', "”": '"', "’": "'", "—": "-", "–": "-"
})

def normalize_and_flatten(txt: str) -> str:
    if not txt:
        return ""
    # NFKC + mapa de ligaduras/pontuação
    txt = unicodedata.normalize("NFKC", txt).translate(LIG_MAP)
    # Remove caracteres invisíveis e NBSPs
    for z in ZW:
        txt = txt.replace(z, "")
    for s in NBSPS:
        txt = txt.replace(s, " ")
    # Des-hifeniza quebras de linha com '-'
    txt = re.sub(r"-\s*\n\s*", "", txt)
    # Normaliza variações de "Nº"
    txt = re.sub(r"(?<!\w)N(?:º|°|o|\.)\b\s*:?", "Nº: ", txt, flags=re.I)
    # Achata quebras de linha e múltiplos espaços
    txt = re.sub(r"\s*\n+\s*", " ", txt)
    txt = re.sub(r"[ \t]{2,}", " ", txt).strip()
    return txt

def _tesseract_ok() -> bool:
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False

# -----------------------------------------------------------------------------
# Renderização e OCR
# -----------------------------------------------------------------------------

def _render_page_image(pdf_bytes: bytes, idx0: int, dpi: int) -> Image.Image:
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        page = doc[idx0]
        mat = fitz.Matrix(dpi/72.0, dpi/72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

def _ocr_grey(img: Image.Image, psm: int, lang: str) -> str:
    g = ImageOps.autocontrast(ImageOps.grayscale(img))
    return pytesseract.image_to_string(g, lang=lang, config=f"--oem 1 --psm {psm}")

def _ocr_binary(img: Image.Image, thr: int, psm: int, lang: str) -> str:
    g = ImageOps.grayscale(img)
    g = g.point(lambda x: 255 if x > thr else 0, mode='L')
    return pytesseract.image_to_string(g, lang=lang, config=f"--oem 1 --psm {psm}")

def ocr_page_from_pdf(pdf_bytes: bytes, idx0: int, dpi: int = 350, lang: str = "por") -> str:
    if not _tesseract_ok():
        raise RuntimeError("Tesseract ausente.")
    img = _render_page_image(pdf_bytes, idx0, dpi)
    t1 = _ocr_grey(img, psm=6, lang=lang)
    if len(normalize_and_flatten(t1)) >= 30:
        return t1
    t2 = _ocr_grey(img, psm=4, lang=lang)
    if len(normalize_and_flatten(t2)) >= 30:
        return t2
    return t2

def ocr_page_targeted(pdf_bytes: bytes, idx0: int) -> Tuple[str, Optional[Dict]]:
    if not _tesseract_ok():
        return "", None
    for dpi in (550, 600, 500):
        img = _render_page_image(pdf_bytes, idx0, dpi)
        img = ImageOps.autocontrast(img).filter(
            ImageFilter.UnsharpMask(radius=1.0, percent=120, threshold=3)
        )
        cfgs = [("gray", 6, "por+eng"), ("gray", 4, "por+eng")]
        for mode, psm, lang in cfgs:
            tx = _ocr_grey(img, psm=psm, lang=lang)
            if len(normalize_and_flatten(tx)) >= 60:
                return tx, {"dpi": dpi, "psm": psm, "lang": lang, "mode": mode}
        txb = _ocr_binary(img, thr=170, psm=6, lang="por+eng")
        if len(normalize_and_flatten(txb)) >= 60:
            return txb, {"dpi": dpi, "psm": 6, "lang": "por+eng", "mode": "bin[170]"}
    return "", None

# -----------------------------------------------------------------------------
# Leitura de texto com diagnóstico por página
# -----------------------------------------------------------------------------

def read_pymupdf_texts_with_diag(pdf_bytes: bytes) -> Tuple[List[str], List[Tuple[int,int,int]]]:
    texts, diag = [], []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for i, p in enumerate(doc, start=1):
            t = p.get_text() or ""
            n = len(p.get_images(full=True))
            texts.append(t)
            diag.append((len(t.strip()), n, i))
    return texts, diag

# -----------------------------------------------------------------------------
# Âncoras e regex para "Número da Nota"
# -----------------------------------------------------------------------------

ANCHORS_DIRECT = [
    r"\bN[úu]mero\s+da\s+Nota(?:\s+Fiscal)?\b",
    r"\bN[úu]mero\s+da\s+NFS[\-\s]?e\b",
    r"\bN[úu]mero\s+da\s+NF[\-\s]?e\b",
    r"\bNFS[\-\s]?e\s*(?:N[º°o\.]\b|N[úu]mero|Nr\.?|Num\.?)\b",
    r"\bNF[\-\s]?e?\s*(?:N[º°o\.]\b|N[úu]mero|Nr\.?|Num\.?)\b",
    r"\bNota(?:\s+Fiscal)?\s*(?:N[º°o\.]\b|N[úu]mero|Nr\.?|Num\.?)\b",
    r"\bN[º°o\.]\s*(?:da\s+Nota(?:\s+Fiscal)?|NF[\-\s]?e|NFS[\-\s]?e|NF)\b",
    r"\bN[úu]mero\s+NF(?:[\-\s]?e)?\b",
]
ANCHOR_DIRECT_RE = re.compile("|".join(ANCHORS_DIRECT), flags=re.I)

CTX = r"(?:NFS[\-\s]?e|NF[\-\s]?e|Nota(?:\s+Fiscal)(?:\s+de\s+Servi[cç]os)?(?:\s+Eletr[oô]nica)?)"
LABEL = r"(?:Nº|N[úu]mero|Nr\.?|Num\.?)"
ANCHOR_CTX_LABEL_RE = re.compile(
    rf"(?is)\b{CTX}\b[^0-9A-Za-z]{{0,150}}\b({LABEL})\b"
)

NUM_TOKEN = re.compile(r"(?<!\d)(\d{2,4}/\d{1,8}|\d{1,12})(?![0-9A-Za-z/])")

FORBID_RIGHT = re.compile(
    r"(?is)\b("
    r"data(?:\s*e\s*hora)?|hora|emiss[aã]o|compet[eê]ncia|"
    r"c[óo]digo\s*de\s*verifica[cç][aã]o|autenticidade"
    r")\b"
)

LABEL_ONLY_RE = re.compile(r"(?i)\bNº\s*:?\b")

def refine_nota_number(token: str) -> str:
    if "/" in token:
        left, right = token.split("/", 1)
        ld = re.sub(r"\D+", "", left)
        rd = re.sub(r"\D+", "", right)
        if ld and rd and len(ld) in (2, 4):
            try:
                yr = int(ld) if len(ld) == 4 else (2000 + int(ld) if int(ld) <= 50 else 1900 + int(ld))
            except Exception:
                yr = None
            if yr and 1900 <= yr <= 2099:
                return rd
        return token
    return token

def _anchor_spans(flat_text: str) -> List[Tuple[int,int]]:
    spans = []
    for m in ANCHOR_DIRECT_RE.finditer(flat_text):
        spans.append(m.span())
    for m in ANCHOR_CTX_LABEL_RE.finditer(flat_text):
        spans.append(m.span(1))
    spans.sort(key=lambda x: x[0])
    return spans

def find_nota_number_anchored(flat_text: str, win: int = 120) -> Optional[Dict]:
    anchors = _anchor_spans(flat_text)
    if anchors:
        for a_start, a_end in anchors:
            # Direita
            right_slice = flat_text[a_end:a_end+win]
            for m in NUM_TOKEN.finditer(right_slice):
                if FORBID_RIGHT.search(right_slice[:m.start()]):
                    continue
                tok = m.group(1) if m.lastindex else m.group()
                num = refine_nota_number(tok)
                s = a_end + m.start()
                e = a_end + m.end()
                return {"numero": num, "origem": "âncora: direita", "start": s, "end": e}
            # Esquerda
            left_slice = flat_text[max(0, a_start - win):a_start]
            ms = list(NUM_TOKEN.finditer(left_slice))
            if ms:
                m = ms[-1]
                tok = m.group(1) if m.lastindex else m.group()
                num = refine_nota_number(tok)
                base = max(0, a_start - win)
                s = base + m.start()
                e = base + m.end()
                return {"numero": num, "origem": "âncora: esquerda", "start": s, "end": e}
    # Fallback "Nº + contexto"
    for lab in LABEL_ONLY_RE.finditer(flat_text):
        ls, le = lab.span()
        L = max(0, ls - 300)
        R = min(len(flat_text), le + 300)
        ctx = flat_text[L:R]
        if not re.search(rf"(?i){CTX}", ctx):
            continue
        right_slice = flat_text[le:le + max(win, 160)]
        for m in NUM_TOKEN.finditer(right_slice):
            if FORBID_RIGHT.search(right_slice[:m.start()]):
                continue
            tok = m.group(1) if m.lastindex else m.group()
            num = refine_nota_number(tok)
            s = le + m.start()
            e = le + m.end()
            return {"numero": num, "origem": "fallback: Nº + contexto", "start": s, "end": e}
    return None

def postprocess_num(num: str) -> str:
    n = re.sub(r"\D+", "", num)
    if not n:
        return num
    n = n.lstrip("0")
    return n if n else "0"

# -----------------------------------------------------------------------------
# CNPJ: âncoras, regex, validação e busca
# -----------------------------------------------------------------------------

CNPJ_TOKEN_MASKED_RE = re.compile(r"(?:\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b)")
CNPJ_TOKEN_RAW_RE    = re.compile(r"(?<!\d)(\d{14})(?!\d)")

CNPJ_ANCHORS_DIRECT = [
    r"\bCNPJ\b",
    r"\bCNPJ/CPF\b",
    r"\bCNPJ\s*(?:do|da)\s*(?:Emitente|Prestador|Tomador|Fornecedor)\b",
    r"\b(?:Emitente|Prestador(?:\s+de\s+Servi[cç]os)?|Tomador|Fornecedor)\b",
]
CNPJ_ANCHORS_DIRECT_RE = re.compile("|".join(CNPJ_ANCHORS_DIRECT), flags=re.I)
CNPJ_LABEL_ONLY_RE = re.compile(r"(?i)\bCNPJ(?:/CPF)?\b\s*:?\s*")

def _digits_only(s: str) -> str:
    return re.sub(r"\D+", "", s or "")

def is_valid_cnpj(cnpj: str) -> bool:
    n = _digits_only(cnpj)
    if len(n) != 14:
        return False
    if n == n[0] * 14:
        return False

    def dv_calc(nums: str, pesos: List[int]) -> str:
        soma = sum(int(d) * p for d, p in zip(nums, pesos))
        r = soma % 11
        return "0" if r < 2 else str(11 - r)

    pesos1 = [5,4,3,2,9,8,7,6,5,4,3,2]
    pesos2 = [6] + pesos1

    dv1 = dv_calc(n[:12], pesos1)
    dv2 = dv_calc(n[:12] + dv1, pesos2)

    return n[-2:] == dv1 + dv2

def postprocess_cnpj(token: str) -> str:
    n = _digits_only(token)
    if is_valid_cnpj(n):
        return n
    return ""

def _iter_cnpj_tokens(text_slice: str):
    # Com máscara
    for m in CNPJ_TOKEN_MASKED_RE.finditer(text_slice):
        tok = m.group(0)
        n = postprocess_cnpj(tok)
        if n:
            yield (n, m.start(), m.end())
    # 14 dígitos puros
    for m in CNPJ_TOKEN_RAW_RE.finditer(text_slice):
        tok = m.group(1) if m.lastindex else m.group(0)
        n = postprocess_cnpj(tok)
        if n:
            yield (n, m.start(), m.end())

def find_cnpj_anchored(flat_text: str, win: int = 160) -> Optional[Dict]:
    anchors = [m.span() for m in CNPJ_ANCHORS_DIRECT_RE.finditer(flat_text)]
    # 1) Âncoras conhecidas
    if anchors:
        for a_start, a_end in anchors:
            # Direita
            right = flat_text[a_end:a_end + win]
            for n, s, e in _iter_cnpj_tokens(right):
                return {"cnpj": n, "origem_cnpj": "CNPJ: âncora: direita", "start": a_end + s, "end": a_end + e}
            # Esquerda (último válido)
            left = flat_text[max(0, a_start - win):a_start]
            ms = list(_iter_cnpj_tokens(left))
            if ms:
                n, s, e = ms[-1]
                base = max(0, a_start - win)
                return {"cnpj": n, "origem_cnpj": "CNPJ: âncora: esquerda", "start": base + s, "end": base + e}
    # 2) Label "CNPJ:" explícito
    for lab in CNPJ_LABEL_ONLY_RE.finditer(flat_text):
        ls, le = lab.span()
        right = flat_text[le:le + max(win, 200)]
        for n, s, e in _iter_cnpj_tokens(right):
            return {"cnpj": n, "origem_cnpj": "CNPJ: label-only direita", "start": le + s, "end": le + e}
        left = flat_text[max(0, ls - win):ls]
        ms = list(_iter_cnpj_tokens(left))
        if ms:
            n, s, e = ms[-1]
            base = max(0, ls - win)
            return {"cnpj": n, "origem_cnpj": "CNPJ: label-only esquerda", "start": base + s, "end": base + e}
    # 3) Fallback livre: primeiro CNPJ válido no texto
    for n, s, e in _iter_cnpj_tokens(flat_text):
        return {"cnpj": n, "origem_cnpj": "CNPJ: fallback livre", "start": s, "end": e}
    return None

# -----------------------------------------------------------------------------
# Orquestração por página (com logs)
# -----------------------------------------------------------------------------

def process_document_with_page_fallback(
    pdf_bytes: bytes, win: int, ocr_dpi: int, max_pages: Optional[int]
) -> Tuple[List[Dict], List[Dict]]:
    results, logs = [], []
    texts_raw, diag = read_pymupdf_texts_with_diag(pdf_bytes)
    total_pages = len(texts_raw) if max_pages is None else min(max_pages, len(texts_raw))

    for idx in range(total_pages):
        raw = texts_raw[idx]
        pnum = idx + 1
        ln, imgs, _ = diag[idx]
        method_used = "PyMuPDF"

        flat = normalize_and_flatten(raw)
        got = find_nota_number_anchored(flat, win=120)

        if not got and (len(flat) < 60 or imgs >= 1):
            try:
                o = ocr_page_from_pdf(pdf_bytes, idx, dpi=ocr_dpi, lang="por")
                flat = normalize_and_flatten(o)
                got = find_nota_number_anchored(flat, win=120)
                method_used = "OCR" if got else "OCR"
            except Exception as e:
                method_used = f"OCR erro: {e}"

        if not got:
            tx, cfg = ocr_page_targeted(pdf_bytes, idx)
            if tx:
                flat = normalize_and_flatten(tx)
                got = find_nota_number_anchored(flat, win=120)
                if got:
                    method_used = (
                        f"OCR direcionado (dpi={cfg['dpi']}, psm={cfg['psm']}, "
                        f"lang={cfg['lang']}, mode={cfg['mode']})"
                    )

        # --- NOVO: extrair CNPJ no mesmo flat ---
        got_cnpj = find_cnpj_anchored(flat, win=160)

        logs.append({
            "pagina": pnum,
            "metodo_usado": method_used,
            "len_texto_pymupdf": ln,
            "imagens": imgs,
            "fallback_ocr": method_used.startswith("OCR"),
            "achou": bool(got),
            "flat_preview": flat[:1500],
            "achou_cnpj": bool(got_cnpj),
        })

        if got:
            num_fmt = postprocess_num(got["numero"])
            trecho = flat[max(0, got["start"] - 60):min(len(flat), got["end"] + 60)]

            # CNPJ (opcional)
            cnpj_fmt, cnpj_origem, cnpj_trecho = "", "", ""
            if got_cnpj:
                cnpj_fmt = got_cnpj["cnpj"]                    # só dígitos + validado
                cnpj_origem = got_cnpj["origem_cnpj"]
                c_s, c_e = got_cnpj["start"], got_cnpj["end"]
                cnpj_trecho = flat[max(0, c_s - 60):min(len(flat), c_e + 60)]

            results.append({
                "pagina": pnum,
                "numero": num_fmt,
                "origem": got["origem"],
                "modo": method_used,
                "trecho": trecho,
                # --- Campos CNPJ adicionados ---
                "cnpj": cnpj_fmt,
                "origem_cnpj": cnpj_origem,
                "trecho_cnpj": cnpj_trecho
            })

    return results, logs

# -----------------------------------------------------------------------------
# API pública principal
# -----------------------------------------------------------------------------

def extrair_numero_nf(pdf_bytes: bytes, max_pages: int = 3) -> pd.DataFrame:
    res, logs = process_document_with_page_fallback(pdf_bytes, win=120, ocr_dpi=350, max_pages=max_pages)
    if not res:
        # Mantém o schema esperado + novas colunas de CNPJ
        return pd.DataFrame(columns=[
            "Página", "Número", "Origem", "Modo", "Trecho",
            "CNPJ", "Origem CNPJ", "Trecho CNPJ"
        ])
    df = pd.DataFrame(res)
    df = df.rename(columns={
        "pagina": "Página",
        "numero": "Número",
        "origem": "Origem",
        "modo": "Modo",
        "trecho": "Trecho",
        "cnpj": "CNPJ",
        "origem_cnpj": "Origem CNPJ",
        "trecho_cnpj": "Trecho CNPJ"
    })
    return df

# -----------------------------------------------------------------------------
# Stubs/iteradores (preservados)
# -----------------------------------------------------------------------------

def identify_invoices_in_files(files, ocr_if_needed=False, max_pages=None, dedup_across_pages=True):
    raise NotImplementedError

def iter_zip_pdfs(zip_bytes: bytes, ocr_if_needed=False, max_pages=None):
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            if info.filename.lower().endswith(".pdf"):
                yield info.filename, pd.DataFrame()

def iter_dir_pdfs(dir_path: str, ocr_if_needed=False, max_pages=None, recursive=True):
    yield dir_path, pd.DataFrame()