# modulos/extrator_rfg.py
import re, io, json, csv
from typing import Dict, Optional, List
import fitz  # PyMuPDF


def extract_text_from_pdf(file_bytes: bytes) -> str:
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        return "\n".join(page.get_text("text") for page in doc)


def _norm_text(txt: str) -> str:
    for a, b in { "\u00A0":" ", "–":"-", "—":"-", "‑":"-", "“":'"', "”":'"', "’":"'", "‘":"'" }.items():
        txt = txt.replace(a, b)
    txt = re.sub(r"[ \t\f\r]+", " ", txt)
    return re.sub(r"\n{2,}", "\n", txt).strip()


def _norm_cep(v: str) -> str:
    d = re.sub(r"\D", "", v or "")
    return f"{d[:5]}-{d[5:]}" if len(d) == 8 else (v or "").strip()


# ---------- rótulos (linhas puras) ----------
PAT_LOGR = re.compile(r"^\s*LOGRADOURO\s*:?\s*$", re.I)
PAT_NUM_PURO = re.compile(r"^\s*N[ÚU]MERO\s*(?::|-)?\s*$", re.I)     # "NÚMERO" puro
PAT_NUM_INSCR = re.compile(r"^\s*N[ÚU]MERO\s+DE\s+INSCRI", re.I)     # evitar (nº inscrição)
PAT_CEP = re.compile(r"^\s*CEP\s*:?\s*$", re.I)
PAT_MUN = re.compile(r"^\s*MUNIC[IÍ]PIO\s*:?\s*$", re.I)
PAT_UF  = re.compile(r"^\s*UF\s*:?\s*$", re.I)

def _is_label(s: str) -> bool:
    s = (s or "").strip()
    return any(p.match(s) for p in (PAT_LOGR, PAT_NUM_PURO, PAT_CEP, PAT_MUN, PAT_UF)) or bool(PAT_NUM_INSCR.match(s))


def _next_val(lines: List[str], i: int) -> Optional[str]:
    """
    Valor na mesma linha (após ':'/'-') ou na próxima linha não vazia.
    Nunca retorna um rótulo (ex.: 'UF', 'MUNICÍPIO').
    """
    # valor na mesma linha?
    same = re.sub(r"^\s*[^:–\-]+[:\-–]\s*", "", lines[i]).strip()
    if same and not _is_label(same):
        return same

    # ou na próxima linha útil?
    j = i + 1
    while j < len(lines):
        v = lines[j].strip()
        if v and not re.fullmatch(r"[\*\-–\.]+", v) and not _is_label(v):
            return v
        j += 1
    return None


def extract_rfg_fields(raw_text: str) -> Dict[str, Optional[str]]:
    txt = _norm_text(raw_text)
    lines = [ln.strip() for ln in txt.split("\n")]

    logradouro = numero = cep = municipio = uf = None

    # 1) Âncora LOGRADOURO
    i_log = next((i for i, ln in enumerate(lines) if PAT_LOGR.match(ln)), -1)
    if i_log >= 0:
        logradouro = _next_val(lines, i_log)

        # 2) NÚMERO (somente rótulo puro; ignorar "NÚMERO DE INSCRIÇÃO")
        for j in range(i_log + 1, min(i_log + 12, len(lines))):
            ln = lines[j]
            if PAT_NUM_INSCR.match(ln):
                continue
            if PAT_NUM_PURO.match(ln):
                numero = _next_val(lines, j)
                break

        # 3) CEP
        i_cep = next((i for i in range(i_log, len(lines)) if PAT_CEP.match(lines[i])), -1)
        if i_cep >= 0:
            cep = _next_val(lines, i_cep)
            if cep:
                cep = _norm_cep(cep)

        # 4) MUNICÍPIO
        i_mun = next((i for i in range(i_log, len(lines)) if PAT_MUN.match(lines[i])), -1)
        if i_mun >= 0:
            municipio = _next_val(lines, i_mun)

        # 5) UF
        i_uf = next((i for i in range(i_log, len(lines)) if PAT_UF.match(lines[i])), -1)
        if i_uf >= 0:
            v = (_next_val(lines, i_uf) or "").strip().upper()
            m = re.match(r"[A-Z]{2}", v)
            uf = m.group(0) if m else (v or None)

    # 6) Fallback global p/ NÚMERO (se ainda vazio), ainda ignorando "NÚMERO DE INSCRIÇÃO"
    if not numero:
        for i, ln in enumerate(lines):
            if PAT_NUM_INSCR.match(ln):
                continue
            if PAT_NUM_PURO.match(ln):
                numero = _next_val(lines, i)
                if numero:
                    break

    return {
        "Logradouro": logradouro,
        "Número": numero,
        "CEP": cep,
        "Município": municipio,
        "UF": uf,
    }


# ---------- utilitários de exportação ----------
def as_json_bytes(data) -> bytes:
    return json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")


def rows_to_csv_bytes(rows: List[Dict[str, Optional[str]]]) -> bytes:
    cols = ["Arquivo", "Logradouro", "Número", "CEP", "Município", "UF"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, delimiter=";", lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({c: r.get(c, "") for c in cols})
    return buf.getvalue().encode("utf-8-sig")