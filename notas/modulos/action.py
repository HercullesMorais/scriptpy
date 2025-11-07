# action.py
# -----------------------------------------------------------------------------
# Renomeia PDFs de um ZIP com base no balancete.
# Regras de match (configuráveis):
#   - Padrão: prioriza CNPJ + Número (com correções OCR) e mantém ordem do balancete.
#   - TESTE: número-apenas (ignora CNPJ) quando ignore_cnpj=True.
#
# Nome final: "<Fornecedor> + <Número>.pdf"
# Ordem de gravação: segue a ordem do balancete; pode prefixar "001 - " (prefix_order=True).
#
# Assinatura pública (compatível com o app atual):
#   process_zip_rename(
#       zip_bytes, balancete_rows, max_pages_per_pdf=3, keep_unmatched=True,
#       *, prefix_order=True, ignore_cnpj=False
#   ) -> (novo_zip_bytes, df_log)
# -----------------------------------------------------------------------------

from __future__ import annotations
import io
import re
import zipfile
from dataclasses import dataclass
from typing import Iterable, Dict, Tuple, Optional, List

import pandas as pd

# Usa o identify como motor (retorna colunas "Número" e "CNPJ")
from identify import extrair_numero_nf

# ----------------------- Normalizações e utilitários -----------------------

_OCR_SIMILARES = str.maketrans({
    "O": "0", "o": "0",
    "I": "1", "l": "1", "L": "1",
    "B": "8",
    # Descomente se quiser agressivo:
    # "S": "5",
    # "Z": "2",
})

def _fix_ocr_similar(s: str) -> str:
    return (s or "").translate(_OCR_SIMILARES)

def _digits_only(s: str) -> str:
    return re.sub(r"\D+", "", str(s or ""))

def _cnpj_key(cnpj: str) -> str:
    s = _fix_ocr_similar(cnpj)
    d = _digits_only(s)
    return d if len(d) == 14 else ""

def _num_key(n: str) -> str:
    d = _digits_only(n)
    if not d:
        return ""
    d = d.lstrip("0")
    return d if d else "0"

def _sanitize_filename(name: str, max_len: int = 180) -> str:
    name = re.sub(r'[\\/*?:"<>|]+', "_", str(name or "")).strip()
    name = re.sub(r"\s+", " ", name)
    return name[:max_len] if len(name) > max_len else name

def _pad_index(i: int, width: int) -> str:
    return str(i).zfill(width)

# ----------------------------- Balancete -----------------------------------

@dataclass
class LinhaBalancete:
    fornecedor: str
    cnpj_fornecedor: str
    numero_doc: str

def _coerce_balancete_rows(rows: Iterable[Dict]) -> pd.DataFrame:
    """
    Garante colunas mínimas e adiciona:
      - forn_key: fornecedor (texto)
      - cnpj_key: CNPJ (14 dígitos com correção OCR)
      - num_key : número da nota só dígitos
      - idx     : índice (1-based) para preservar ordem do balancete
    """
    df = pd.DataFrame(list(rows))
    req = {"fornecedor", "cnpj_fornecedor", "numero_doc"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Balancete ausente de colunas obrigatórias: {sorted(missing)}")

    df = df.copy()
    df["forn_key"] = df["fornecedor"].astype(str).str.strip()
    df["cnpj_key"] = df["cnpj_fornecedor"].map(_cnpj_key)
    df["num_key"]  = df["numero_doc"].map(_num_key)
    df["idx"] = range(1, len(df) + 1)

    # Mantém linhas sem chaves? Para número-apenas, basta num_key != ""
    df = df[df["num_key"] != ""]
    return df[["idx","fornecedor","cnpj_fornecedor","numero_doc","forn_key","cnpj_key","num_key"]]

# ------------------------------- Indexação ZIP ------------------------------

@dataclass
class PdfEntry:
    name: str
    data: bytes
    cnpjs: List[str]              # CNPJs normalizados (14 dígitos)
    nums: List[str]               # Números normalizados (somente dígitos)
    pairs: List[Tuple[str,str]]   # pares (cnpj, numero) da mesma linha/página

def _extract_identify_index(pdf_bytes: bytes, max_pages: int) -> Tuple[List[str], List[str], List[Tuple[str,str]]]:
    """
    Executa identify.extrair_numero_nf e retorna:
      - lista de CNPJs normalizados (sem duplicatas)
      - lista de Números normalizados (sem duplicatas)
      - pares (cnpj, numero) quando ambos existirem na mesma linha/página
    """
    df = extrair_numero_nf(pdf_bytes, max_pages=max_pages)
    if df is None or df.empty:
        return [], [], []

    cnpjs_raw = df["CNPJ"].astype(str).tolist() if "CNPJ" in df.columns else []
    nums_raw  = df["Número"].astype(str).tolist() if "Número" in df.columns else []

    cnpjs, nums = [], []
    for c in cnpjs_raw:
        ck = _cnpj_key(c)
        if ck:
            cnpjs.append(ck)
    for n in nums_raw:
        nk = _num_key(n)
        if nk:
            nums.append(nk)

    # de-dup preservando ordem
    def _dedup(seq: List[str]) -> List[str]:
        seen = set(); out = []
        for x in seq:
            if x not in seen:
                seen.add(x); out.append(x)
        return out

    cnpjs = _dedup(cnpjs)
    nums  = _dedup(nums)

    pairs: List[Tuple[str,str]] = []
    if "CNPJ" in df.columns and "Número" in df.columns:
        for c, n in zip(df["CNPJ"].tolist(), df["Número"].tolist()):
            ck, nk = _cnpj_key(c), _num_key(n)
            if ck and nk:
                pairs.append((ck, nk))

    return cnpjs, nums, pairs

def _index_zip_pdfs(zip_bytes: bytes, max_pages_per_pdf: int) -> Tuple[List[PdfEntry], List[Tuple[str,str]]]:
    """
    Lê o ZIP uma vez e indexa PDFs com (cnpjs, nums, pairs).
    Também retorna uma lista de erros (arquivo, motivo).
    """
    pdfs: List[PdfEntry] = []
    errors: List[Tuple[str,str]] = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            fn = info.filename
            try:
                data = z.read(fn)
            except Exception as e:
                errors.append((fn, f"read_error: {e}"))
                continue

            if not fn.lower().endswith(".pdf"):
                pdfs.append(PdfEntry(name=fn, data=data, cnpjs=[], nums=[], pairs=[]))
                continue

            try:
                cnpjs, nums, pairs = _extract_identify_index(data, max_pages=max_pages_per_pdf)
                pdfs.append(PdfEntry(name=fn, data=data, cnpjs=cnpjs, nums=nums, pairs=pairs))
            except Exception as e:
                errors.append((fn, f"identify_error: {e}"))
                pdfs.append(PdfEntry(name=fn, data=data, cnpjs=[], nums=[], pairs=[]))

    return pdfs, errors

# ------------------------------- Matching ----------------------------------

def _match_bal_line_to_pdf(
    bal_row: pd.Series,
    candidates: List[PdfEntry],
    used: set,
    ignore_cnpj: bool
) -> Tuple[Optional[int], str]:
    """
    Retorna (index_pdf, modo):
      - modo = "pair", "set", "num_only"
    Estratégias:
      1) (default) par (cnpj,num) exato
      2) (default) ambos nos conjuntos (cnpj in cnpjs AND num in nums)
      3) (quando ignore_cnpj=True) número-apenas
    """
    nk = bal_row["num_key"]
    cj = bal_row["cnpj_key"]

    if not ignore_cnpj:
        # 1) Pareamento forte (par exato)
        for idx, pe in enumerate(candidates):
            if idx in used: continue
            if (cj, nk) in pe.pairs:
                return idx, "pair"
        # 2) Conjuntos (CNPJ e Número presentes, ainda que não no mesmo par)
        for idx, pe in enumerate(candidates):
            if idx in used: continue
            if (cj in pe.cnpjs) and (nk in pe.nums):
                return idx, "set"

    # 3) Número-apenas (TESTE)
    for idx, pe in enumerate(candidates):
        if idx in used: continue
        if nk in pe.nums:
            return idx, "num_only"

    return None, ""

# ---------------------------- Escrita do ZIP --------------------------------

def _unique_name_in_zip(zout: zipfile.ZipFile, written: set, desired: str) -> str:
    """Garante nome único no ZIP final."""
    base, ext = desired, ""
    if "." in desired:
        p = desired.rfind(".")
        base, ext = desired[:p], desired[p:]
    candidate = desired
    k = 1
    existing = set(zout.namelist()) | set(written)
    while candidate in existing:
        candidate = f"{base} ({k}){ext}"
        k += 1
    return candidate

# ---------------------------- Função principal ------------------------------

def process_zip_rename(
    zip_bytes: bytes,
    balancete_rows: Iterable[Dict],
    max_pages_per_pdf: int = 3,
    keep_unmatched: bool = True,
    *,
    prefix_order: bool = True,
    ignore_cnpj: bool = False  # <<< TESTE: defina True para número-apenas
) -> Tuple[bytes, pd.DataFrame]:
    """
    Processa o ZIP:
      - Indexa PDFs (uma passada) com (cnpjs, nums, pairs)
      - Percorre o balancete NA ORDEM e casa com a estratégia solicitada
      - Grava os renomeados na ordem do balancete
      - Copia não-casados (e não-PDFs) se keep_unmatched=True
    Retorno: (zip_bytes_final, df_log)
    """
    df_bal = _coerce_balancete_rows(balancete_rows)
    pdfs, idx_errors = _index_zip_pdfs(zip_bytes, max_pages_per_pdf)

    buf_out = io.BytesIO()
    log_rows = []
    written = set()
    pad_w = max(3, len(str(len(df_bal))))

    with zipfile.ZipFile(buf_out, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        used_pdf_idx: set = set()

        # 1) Gravar MATCHES na ordem do balancete
        for _, row in df_bal.sort_values("idx").iterrows():
            m_idx, modo = _match_bal_line_to_pdf(row, pdfs, used_pdf_idx, ignore_cnpj=ignore_cnpj)
            if m_idx is None:
                log_rows.append({
                    "arquivo_original": "",
                    "status": "BAL_SEM_MATCH",
                    "novo_nome": "",
                    "numero": row["numero_doc"],
                    "fornecedor": row["fornecedor"],
                    "cnpj": row["cnpj_fornecedor"],
                    "detalhe": "sem PDF compatível (use ignore_cnpj=True para teste número-apenas)" if not ignore_cnpj else "sem PDF compatível por número"
                })
                continue

            used_pdf_idx.add(m_idx)
            pe = pdfs[m_idx]

            # Nome final: "<Fornecedor> + <Número>.pdf"
            base_name = f"{row['fornecedor']} + {row['numero_doc']}.pdf"
            if prefix_order:
                base_name = f"{_pad_index(row['idx'], pad_w)} - {base_name}"

            out_name = _sanitize_filename(base_name)
            out_name = _unique_name_in_zip(zout, written, out_name)
            zout.writestr(out_name, pe.data)
            written.add(out_name)

            log_rows.append({
                "arquivo_original": pe.name,
                "status": "RENOMEADO",
                "novo_nome": out_name,
                "numero": row["numero_doc"],
                "fornecedor": row["fornecedor"],
                "cnpj": row["cnpj_fornecedor"],
                "detalhe": f"modo={modo} pairs_pdf={pe.pairs} cnpjs_pdf={pe.cnpjs} nums_pdf={pe.nums}"
            })

        # 2) Copiar demais arquivos não usados (se keep_unmatched=True)
        if keep_unmatched:
            for i, pe in enumerate(pdfs):
                if i in used_pdf_idx:
                    continue
                out_name = _unique_name_in_zip(zout, written, pe.name)
                zout.writestr(out_name, pe.data)
                written.add(out_name)
                status = "IGNORADO (não-PDF)" if not pe.name.lower().endswith(".pdf") else "SEM_MATCH"
                log_rows.append({
                    "arquivo_original": pe.name,
                    "status": status,
                    "novo_nome": out_name if status != "SEM_MATCH" else "",
                    "numero": "",
                    "fornecedor": "",
                    "cnpj": "",
                    "detalhe": f"pairs_pdf={pe.pairs} cnpjs_pdf={pe.cnpjs} nums_pdf={pe.nums}" if status == "SEM_MATCH" else ""
                })

        # 3) Erros de indexação
        for fn, err in idx_errors:
            log_rows.append({
                "arquivo_original": fn,
                "status": "ERRO_INDEXACAO",
                "novo_nome": "",
                "numero": "",
                "fornecedor": "",
                "cnpj": "",
                "detalhe": err
            })

    buf_out.seek(0)
    df_log = pd.DataFrame(log_rows, columns=[
        "arquivo_original", "status", "novo_nome", "numero", "fornecedor", "cnpj", "detalhe"
    ])
    return buf_out.getvalue(), df_log