# modulos/extrator_parecer.py
import re, io, json, csv
from typing import Dict, Optional, List
import fitz  # PyMuPDF


# -----------------------------
# Leitura e normalização
# -----------------------------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extrai texto de todas as páginas do PDF (modo 'text' do PyMuPDF)."""
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        return "\n".join(page.get_text("text") for page in doc)


def _norm_text(txt: str) -> str:
    """Normaliza espaços e pontuações para facilitar regex em PDFs."""
    for a, b in {
        "\u00A0":" ", "–":"-", "—":"-", "‑":"-",
        "“":'"', "”":'"', "’":"'", "‘":"'"
    }.items():
        txt = txt.replace(a, b)
    txt = re.sub(r"[ \t\f\r]+", " ", txt)
    txt = re.sub(r"\n{2,}", "\n", txt)
    return txt.strip()


# -----------------------------
# Regras de extração
# -----------------------------
def _mask_cnpj(raw: str) -> str:
    d = re.sub(r"\D", "", raw or "")
    return f"{d[:2]}.{d[2:5]}.{d[5:8]}/{d[8:12]}-{d[12:]}" if len(d) == 14 else (raw or "").strip()


def extract_cnpj(text: str) -> Optional[str]:
    """
    Captura CNPJ após 'CNPJ' (ou 'CNPJ/MF'), tolerando variações de máscara.
    Ex.: CNPJ: 02.435.301/0001-73
    """
    m = re.search(
        r"\bCNPJ(?:\/MF)?\s*[:\-]?\s*([0-9]{2}\.?[0-9]{3}\.?[0-9]{3}[\/]?[0-9]{4}\-?[0-9]{2})",
        text, flags=re.IGNORECASE
    )
    return _mask_cnpj(m.group(1)) if m else None


def extract_company_name(text: str) -> Optional[str]:
    """
    Padrão principal:
      '... apresentada pela empresa NOME, CNPJ:'
      'empresa "NOME", CNPJ ...'
    Fallback: último trecho significativo antes de 'CNPJ'.
    """
    m = re.search(
        r"empresa\s+\"?(.{3,150}?)\"?\s*,\s*CNPJ\b",
        text, flags=re.IGNORECASE | re.DOTALL
    )
    if m:
        name = re.sub(r"\s+", " ", m.group(1)).strip(" ,.;:-")
        return name

    m_cnpj = re.search(r"\bCNPJ\b", text, flags=re.IGNORECASE)
    if m_cnpj:
        start = max(0, m_cnpj.start() - 160)
        snippet = text[start:m_cnpj.start()]
        # tenta razão social com sufixos societários comuns
        candidates = re.findall(
            r"([A-Z0-9][A-Z0-9 &\.\-/ÁÉÍÓÚÂÊÔÃÕÇ]{3,}?(?:\s+(?:LTDA|EIRELI|S\.?A\.?|ME|EPP|MEI|S\/A)))",
            snippet, flags=re.IGNORECASE
        )
        if candidates:
            return re.sub(r"\s+", " ", candidates[-1]).strip(" ,.;:-")

        parts = [p.strip() for p in re.split(r"[,.;:\n]", snippet) if len(p.strip()) > 3]
        if parts:
            return parts[-1]
    return None


def extract_parecer_contestacao(text: str) -> Optional[str]:
    """
    PARECER TÉCNICO DA CONTESTAÇÃO nº 12873/2024
    Aceita: nº / n° / no ; com/sem espaço.
    """
    m = re.search(
        r"PARECER\s+T[ÉE]CNICO\s+DA\s+CONTESTA[ÇC][ÃA]O\s*n[º°o]?\s*([0-9]{3,7}\s*/\s*\d{4})",
        text, flags=re.IGNORECASE
    )
    if not m:
        return None
    val = re.sub(r"\s*", "", m.group(1))
    return f"{val[:-4]}/{val[-4:]}" if "/" not in val else val


def extract_parecer_tecnico(text: str, parecer_cont: Optional[str]) -> Optional[str]:
    """
    'Parecer Técnico nº 20129/2023' (não confundir com o da Contestação).
    Coberturas:
      - nº / n° / no, com/sem espaço (ex.: 'nº20129/2023');
      - 'Parecer Técnico ... SETEC/MCTI ... nº 20129/2023';
      - evita matches onde apareça 'Contestação' em ~80 chars após 'Técnico'.
    """
    pat = re.compile(
        r"Parecer\s*T[ée]cnico(?![^.\n]{0,80}Contestação)[^.:\n]{0,120}?"
        r"\bn[º°o]?\s*([0-9]{3,7}\s*/\s*\d{4})",
        flags=re.IGNORECASE
    )
    for m in pat.finditer(text):
        raw = re.sub(r"\s*", "", m.group(1))
        norm = f"{raw[:-4]}/{raw[-4:]}" if "/" not in raw else raw
        if not parecer_cont or norm != parecer_cont:
            return norm
    return None


def extract_ano_base(text: str) -> Optional[int]:
    """Captura 'Ano-Base 2021' ou 'Ano Base 2021' (opcional)."""
    m = re.search(r"Ano\s*-?\s*Base\s*(\d{4})", text, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def extract_fields(raw_text: str) -> Dict[str, Optional[str]]:
    """
    Retorna o dicionário com os campos principais extraídos do PDF do Parecer.
    Chaves: 'Nome da empresa', 'CNPJ', 'Nº do Parecer Técnico da Contestação', 'Nº do Parecer Técnico', 'Ano-Base'
    """
    txt = _norm_text(raw_text)
    cnpj = extract_cnpj(txt)
    nome = extract_company_name(txt)
    parecer_cont = extract_parecer_contestacao(txt)
    parecer_tecnico = extract_parecer_tecnico(txt, parecer_cont)
    ano_base = extract_ano_base(txt)

    return {
        "Nome da empresa": nome,
        "CNPJ": cnpj,
        "Nº do Parecer Técnico da Contestação": parecer_cont,
        "Nº do Parecer Técnico": parecer_tecnico,
        "Ano-Base": ano_base,
    }


# -----------------------------
# Utilitários de exportação (compat)
# -----------------------------
def as_json_bytes(data) -> bytes:
    return json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")


def rows_to_csv_bytes(rows: List[Dict[str, Optional[str]]]) -> bytes:
    cols = [
        "Arquivo",
        "Nome da empresa",
        "CNPJ",
        "Nº do Parecer Técnico da Contestação",
        "Nº do Parecer Técnico",
        "Ano-Base",
    ]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, delimiter=";", lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({c: r.get(c, "") for c in cols})
    return buf.getvalue().encode("utf-8-sig")