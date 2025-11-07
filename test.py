# -*- coding: utf-8 -*-
import re, io, zipfile, unicodedata
from typing import List, Tuple, Optional, Dict
import fitz
from PIL import Image, ImageOps, ImageFilter
import pytesseract
import streamlit as st
import pandas as pd

ZW = ["\u200b","\u200c","\u200d","\u2060","\ufeff"]
NBSPS = ["\u00A0","\u2007","\u202F"]
LIG_MAP = str.maketrans({"ﬁ":"fi","ﬂ":"fl","ﬃ":"ffi","ﬄ":"ffl","“":'"',"”":'"',"’":"'", "—":"-","–":"-"})

def normalize_and_flatten(txt: str) -> str:
    if not txt: return ""
    txt = unicodedata.normalize("NFKC", txt).translate(LIG_MAP)
    for z in ZW: txt = txt.replace(z, "")
    for s in NBSPS: txt = txt.replace(s, " ")
    txt = re.sub(r"-\s*\n\s*","",txt)
    txt = re.sub(r"(?<!\w)N(?:º|°|o|\.)\b\s*:?","Nº: ",txt,flags=re.I)
    txt = re.sub(r"\s*\n+\s*"," ",txt)
    txt = re.sub(r"[ \t]{2,}"," ",txt).strip()
    return txt

def _tesseract_ok() -> bool:
    try: pytesseract.get_tesseract_version(); return True
    except: return False

def _render_page_image(pdf_bytes: bytes, idx0: int, dpi: int):
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
    if not _tesseract_ok(): raise RuntimeError("Tesseract ausente.")
    img = _render_page_image(pdf_bytes, idx0, dpi)
    t1 = _ocr_grey(img, psm=6, lang=lang)
    if len(normalize_and_flatten(t1)) >= 30: return t1
    t2 = _ocr_grey(img, psm=4, lang=lang)
    if len(normalize_and_flatten(t2)) >= 30: return t2
    return t2

def ocr_page_targeted(pdf_bytes: bytes, idx0: int) -> Tuple[str, Optional[Dict]]:
    if not _tesseract_ok(): return "", None
    for dpi in (550, 600, 500):
        img = _render_page_image(pdf_bytes, idx0, dpi)
        img = ImageOps.autocontrast(img).filter(ImageFilter.UnsharpMask(radius=1.0, percent=120, threshold=3))
        cfgs = [("gray", 6, "por+eng"), ("gray", 4, "por+eng")]
        for mode, psm, lang in cfgs:
            tx = _ocr_grey(img, psm=psm, lang=lang)
            if len(normalize_and_flatten(tx)) >= 60:
                return tx, {"dpi": dpi, "psm": psm, "lang": lang, "mode": mode}
        txb = _ocr_binary(img, thr=170, psm=6, lang="por+eng")
        if len(normalize_and_flatten(txb)) >= 60:
            return txb, {"dpi": dpi, "psm": 6, "lang": "por+eng", "mode": "bin[170]"}
    return "", None

def read_pymupdf_texts_with_diag(pdf_bytes: bytes) -> Tuple[List[str], List[Tuple[int,int,int]]]:
    texts, diag = [], []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for i, p in enumerate(doc, start=1):
            t = p.get_text() or ""
            n = len(p.get_images(full=True))
            texts.append(t); diag.append((len(t.strip()), n, i))
    return texts, diag

ANCHORS_DIRECT = [
    r"\bN[úu]mero\s+da\s+Nota(?:\s+Fiscal)?\b",
    r"\bN[úu]mero\s+da\s+NFS[\-\s]?e\b",
    r"\bN[úu]mero\s+da\s+NF[\-\s]?e\b",
    r"\bNFS[\-\s]?e\s*(?:N[º°o\.]|N[úu]mero|Nr\.?|Num\.?)\b",
    r"\bNF[\-\s]?e?\s*(?:N[º°o\.]|N[úu]mero|Nr\.?|Num\.?)\b",
    r"\bNota(?:\s+Fiscal)?\s*(?:N[º°o\.]|N[úu]mero|Nr\.?|Num\.?)\b",
    r"\bN[º°o\.]\s*(?:da\s+Nota(?:\s+Fiscal)?|NF[\-\s]?e|NFS[\-\s]?e|NF)\b",
    r"\bN[úu]mero\s+NF(?:[\-\s]?e)?\b",
]
ANCHOR_DIRECT_RE = re.compile("|".join(ANCHORS_DIRECT), flags=re.I)

CTX = r"(?:NFS[\-\s]?e|NF[\-\s]?e|Nota(?:\s+Fiscal)(?:\s+de\s+Servi[cç]os)?(?:\s+Eletr[oô]nica)?)"
LABEL = r"(?:Nº|N[úu]mero|Nr\.?|Num\.?)"
ANCHOR_CTX_LABEL_RE = re.compile(rf"(?is)\b{CTX}\b[^0-9A-Za-z]{{0,150}}\b({LABEL})\b")

NUM_TOKEN = re.compile(r"(?<!\d)(\d{2,4}/\d{1,8}|\d{1,12})(?![0-9A-Za-z/])")
FORBID_RIGHT = re.compile(r"(?is)\b(data(?:\s*e\s*hora)?|hora|emiss[aã]o|compet[eê]ncia|c[óo]digo\s*de\s*verifica[cç][aã]o|autenticidade)\b")
LABEL_ONLY_RE = re.compile(r"(?i)\bNº\s*:?\b")

def refine_nota_number(token: str) -> str:
    if "/" in token:
        left, right = token.split("/",1)
        ld = re.sub(r"\D+","",left); rd = re.sub(r"\D+","",right)
        if ld and rd and len(ld) in (2,4):
            try:
                yr = int(ld) if len(ld)==4 else (2000+int(ld) if int(ld)<=50 else 1900+int(ld))
            except: yr = None
            if yr and 1900<=yr<=2099: return rd
    return token

def _anchor_spans(flat_text: str) -> List[Tuple[int,int]]:
    spans=[]
    for m in ANCHOR_DIRECT_RE.finditer(flat_text): spans.append(m.span())
    for m in ANCHOR_CTX_LABEL_RE.finditer(flat_text): spans.append(m.span(1))
    spans.sort(key=lambda x:x[0])
    return spans

def find_nota_number_anchored(flat_text: str, win: int = 120) -> Optional[Dict]:
    anchors = _anchor_spans(flat_text)
    if anchors:
        for a_start, a_end in anchors:
            right_slice = flat_text[a_end:a_end+win]
            for m in NUM_TOKEN.finditer(right_slice):
                if FORBID_RIGHT.search(right_slice[:m.start()]): continue
                tok = m.group(1) if m.lastindex else m.group()
                num = refine_nota_number(tok)
                s = a_end+m.start(); e = a_end+m.end()
                return {"numero":num,"origem":"âncora: direita","start":s,"end":e}
            left_slice = flat_text[max(0,a_start-win):a_start]
            ms = list(NUM_TOKEN.finditer(left_slice))
            if ms:
                m = ms[-1]
                tok = m.group(1) if m.lastindex else m.group()
                num = refine_nota_number(tok)
                s = max(0,a_start-win)+m.start(); e = max(0,a_start-win)+m.end()
                return {"numero":num,"origem":"âncora: esquerda","start":s,"end":e}
    else:
        for lab in LABEL_ONLY_RE.finditer(flat_text):
            ls, le = lab.span()
            L = max(0, ls-300); R = min(len(flat_text), le+300)
            ctx = flat_text[L:R]
            if not re.search(rf"(?i){CTX}", ctx): 
                continue
            right_slice = flat_text[le:le+max(win,160)]
            for m in NUM_TOKEN.finditer(right_slice):
                if FORBID_RIGHT.search(right_slice[:m.start()]): continue
                tok = m.group(1) if m.lastindex else m.group()
                num = refine_nota_number(tok)
                s = le+m.start(); e = le+m.end()
                return {"numero":num,"origem":"fallback: Nº + contexto","start":s,"end":e}
    return None

def postprocess_num(num: str) -> str:
    n = re.sub(r"\D+","", num)
    if not n: return num
    n = n.lstrip("0")
    return n if n else "0"

def process_document_with_page_fallback(pdf_bytes: bytes, win: int, ocr_dpi: int) -> Tuple[List[Dict], List[Dict]]:
    results, logs = [], []
    texts_raw, diag = read_pymupdf_texts_with_diag(pdf_bytes)
    for idx, raw in enumerate(texts_raw):
        pnum = idx+1
        ln, imgs, _ = diag[idx]
        method_used = "PyMuPDF"
        flat = normalize_and_flatten(raw)
        got = find_nota_number_anchored(flat, win=win)
        if not got and (len(flat)<60 or imgs>=1):
            try:
                o = ocr_page_from_pdf(pdf_bytes, idx, dpi=ocr_dpi, lang="por")
                flat = normalize_and_flatten(o)
                got = find_nota_number_anchored(flat, win=win)
                method_used = "OCR" if got else "OCR"
            except Exception as e:
                method_used = f"OCR erro: {e}"
        if not got:
            tx, cfg = ocr_page_targeted(pdf_bytes, idx)
            if tx:
                flat = normalize_and_flatten(tx)
                got = find_nota_number_anchored(flat, win=win)
                if got:
                    method_used = f"OCR direcionado (dpi={cfg['dpi']}, psm={cfg['psm']}, lang={cfg['lang']}, mode={cfg['mode']})"
        logs.append({
            "pagina": pnum,
            "metodo_usado": method_used,
            "len_texto_pymupdf": ln,
            "imagens": imgs,
            "fallback_ocr": method_used.startswith("OCR"),
            "achou": bool(got),
            "flat_preview": flat[:1500]
        })
        if got:
            num_fmt = postprocess_num(got["numero"])
            trecho = flat[max(0,got["start"]-60):min(len(flat),got["end"]+60)]
            results.append({"numero":num_fmt,"pagina":pnum,"origem":got["origem"],"trecho":trecho,"modo":method_used})
    return results, logs

def iter_zip_pdfs(zip_bytes: bytes):
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        for info in z.infolist():
            if info.is_dir(): continue
            if info.filename.lower().endswith(".pdf"):
                try: yield info.filename, z.read(info)
                except: continue

st.set_page_config(page_title="Número da Nota (PDF→Texto/OCR)", page_icon="🔢", layout="centered")
st.title("Número da Nota (NFS‑e / NF‑e)")

c1,c2 = st.columns(2)
with c1: ocr_dpi = st.slider("DPI OCR (leve)",250,450,350,25)
with c2: win = st.slider("Janela âncora",40,220,120,10)

st.subheader("PDF único")
uploaded = st.file_uploader("Selecione um PDF", type=["pdf"], key="single_pdf")
if uploaded is not None:
    pdf_bytes = uploaded.read()
    with st.status("Processando...", expanded=False):
        try: resultados, logs = process_document_with_page_fallback(pdf_bytes, win=win, ocr_dpi=ocr_dpi)
        except Exception as e: st.error(f"Erro: {e}"); st.stop()

    with st.expander("Diagnóstico"):
        for lg in logs:
            st.text(
                f"Pág.{lg['pagina']:02d} | método: {lg['metodo_usado']} | texto(PyMuPDF): {lg['len_texto_pymupdf']} "
                f"| imgs: {lg['imagens']} | fallback OCR: {lg['fallback_ocr']} | achou: {lg['achou']}"
            )

    with st.expander("Prévia do texto por página"):
        for lg in logs:
            st.code(f"[Página {lg['pagina']}] método={lg['metodo_usado']}\n{lg['flat_preview']}", language="markdown")

    if resultados:
        r0 = resultados[0]
        st.success(f"Número: {r0['numero']} | pág {r0['pagina']} | origem {r0['origem']} | modo {r0['modo']}")
        st.code(r0["trecho"], language="markdown")
        if len(resultados)>1:
            with st.expander("Outras ocorrências"):
                for r in resultados:
                    st.markdown(f"- **{r['numero']}** — pág {r['pagina']} — {r['origem']} — {r['modo']}")
    else:
        st.warning("Não encontrado.")

st.markdown("---")
st.header("ZIP (lote)")
uploaded_zip = st.file_uploader("Selecione um ZIP com PDFs", type=["zip"], key="zip_uploader")
if uploaded_zip is not None:
    zip_bytes = uploaded_zip.read()
    entries = list(iter_zip_pdfs(zip_bytes))
    if not entries: st.error("ZIP sem PDFs.")
    else:
        rows, errs = [], []
        prog = st.progress(0, text="Processando...")
        total = len(entries)
        for i,(name,pdf_b) in enumerate(entries, start=1):
            try:
                res, logs = process_document_with_page_fallback(pdf_b, win=win, ocr_dpi=ocr_dpi)
                if res:
                    for r in res:
                        rows.append({"arquivo":name,"status":"Encontrado","numero":r["numero"],"pagina":r["pagina"],"origem":r["origem"],"modo":r["modo"],"trecho":r["trecho"]})
                else:
                    rows.append({"arquivo":name,"status":"Não encontrado","numero":"","pagina":"","origem":"","modo":"","trecho":""})
            except Exception as e:
                errs.append((name,str(e)))
                rows.append({"arquivo":name,"status":"Erro","numero":"","pagina":"","origem":"","modo":"","trecho":f"Erro: {e}"})
            prog.progress(i/total, text=f"Processando... ({i}/{total})")
        df = pd.DataFrame(rows, columns=["arquivo","status","numero","pagina","origem","modo","trecho"])
        st.dataframe(df, use_container_width=True, height=420)
        st.caption(f"Encontrados: {(df['status']=='Encontrado').sum()} | Não encontrados: {(df['status']=='Não encontrado').sum()} | Erros: {(df['status']=='Erro').sum()}")
        st.download_button("Baixar CSV", data=df.to_csv(index=False).encode("utf-8-sig"), file_name="numeros_notas_lote.csv", mime="text/csv", use_container_width=True)
        if errs:
            with st.expander("Erros"):
                for n,m in errs: st.write(f"- {n} → {m}")