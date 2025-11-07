# -*- coding: utf-8 -*-
"""
Módulo: extrator
- Extrai "Nome Empresarial" e "UF" do PDF (Comprovante CNPJ)
- Normaliza o nome (title case com exceções + correções de acento comuns)
- Converte UF -> Estado por extenso
"""

from __future__ import annotations
import re
from typing import Dict, Optional
import fitz  # PyMuPDF


# Mapa UF -> Estado por extenso
_UF_TO_ESTADO: Dict[str, str] = {
    "AC": "Acre",
    "AL": "Alagoas",
    "AP": "Amapá",
    "AM": "Amazonas",
    "BA": "Bahia",
    "CE": "Ceará",
    "DF": "Distrito Federal",
    "ES": "Espírito Santo",
    "GO": "Goiás",
    "MA": "Maranhão",
    "MT": "Mato Grosso",
    "MS": "Mato Grosso do Sul",
    "MG": "Minas Gerais",
    "PA": "Pará",
    "PB": "Paraíba",
    "PR": "Paraná",
    "PE": "Pernambuco",
    "PI": "Piauí",
    "RJ": "Rio de Janeiro",
    "RN": "Rio Grande do Norte",
    "RS": "Rio Grande do Sul",
    "RO": "Rondônia",
    "RR": "Roraima",
    "SC": "Santa Catarina",
    "SP": "São Paulo",
    "SE": "Sergipe",
    "TO": "Tocantins",
}

# Correções de acentos comuns (best-effort)
_ACENTO_CORRECOES = {
    "Administracao": "Administração",
    "Agencia": "Agência",
    "Comercio": "Comércio",
    "Construcao": "Construção",
    "Educacao": "Educação",
    "Eletrica": "Elétrica",
    "Eletronica": "Eletrônica",
    "Informacao": "Informação",
    "Informacoes": "Informações",
    "Instalacao": "Instalação",
    "Instalacoes": "Instalações",
    "Inteligencia": "Inteligência",
    "Logistica": "Logística",
    "Publicacao": "Publicação",
    "Publicacoes": "Publicações",
    "Solucoes": "Soluções",
    "Industria": "Indústria",
    "Oficios": "Ofícios",
}

def _texto_pdf(pdf_bytes: bytes) -> str:
    """Extrai texto das páginas do PDF (modo 'text', suficiente p/ CNPJ)."""
    if not pdf_bytes:
        return ""
    out = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            out.append(page.get_text("text"))
    return "\n".join(out)

def _capturar_primeiro(texto: str, padroes: list[str]) -> Optional[str]:
    for p in padroes:
        m = re.search(p, texto, flags=re.IGNORECASE)
        if m:
            valor = m.group(1).strip()
            valor = re.sub(r"[\*\u200b]+", "", valor)
            valor = re.sub(r"\s+", " ", valor).strip()
            if valor:
                return valor
    return None

def _parse_nome_empresarial(texto: str) -> Optional[str]:
    padroes = [
        r"NOME\s+EMPRESARIAL\s*[:\-]*\s*([^\n\r]+)",
        r"Nome\s+Empresarial\s*[:\-]*\s*([^\n\r]+)",
        r"RAZAO\s+SOCIAL\s*[:\-]*\s*([^\n\r]+)",
        r"Raz[aã]o\s+Social\s*[:\-]*\s*([^\n\r]+)",
    ]
    return _capturar_primeiro(texto, padroes)

def _parse_uf(texto: str) -> Optional[str]:
    padroes = [
        r"\bUF\b\s*[:\-]?\s*([A-Z]{2})\b",
        r"\bEstado\b\s*[:\-]?\s*([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ ]+)\b",
    ]
    valor = _capturar_primeiro(texto, padroes)
    if not valor:
        return None
    valor = valor.strip()
    if len(valor) > 2:  # já por extenso?
        alvo = valor.lower()
        for uf, nome in _UF_TO_ESTADO.items():
            if nome.lower() == alvo:
                return uf
        return None
    return valor.upper()

def _normalizar_nome_empresa(raw: str) -> str:
    """Title case com exceções, correções de acentos e sufixos societários em CAIXA ALTA."""
    if not raw:
        return ""
    nome = re.sub(r"\s+", " ", raw.strip())
    nome = nome.lower().title()

    # preposições em minúsculas (exceto nas pontas)
    exc_lower = {"de", "da", "do", "das", "dos", "e", "para", "por", "a", "as", "o", "os"}
    partes = nome.split()
    for i, w in enumerate(partes):
        if w.lower() in exc_lower and 0 < i < len(partes) - 1:
            partes[i] = w.lower()
    nome = " ".join(partes)

    # Correções de acento comuns
    for sem, com in _ACENTO_CORRECOES.items():
        nome = re.sub(rf"\b{sem}\b", com, nome)

    # Sufixos societários em CAIXA ALTA
    sufixos_upper = {
        "LTDA": {"Ltda", "ltda"},
        "EIRELI": {"Eireli", "eireli"},
        "ME": {"Me", "me"},
        "MEI": {"Mei", "mei"},
        "EPP": {"Epp", "epp"},
        "S.A.": {"S.a.", "s.a.", "S.A", "s.a", "SA", "Sa"},
        "S/A": {"S/a", "s/a"},
        "SPE": {"Spe", "spe"},
    }
    for alvo, variantes in sufixos_upper.items():
        for v in variantes:
            nome = re.sub(rf"\b{re.escape(v)}\b", alvo, nome)

    # Tratar "S A" (com espaço) como S.A.
    nome = re.sub(r"\bS\s*A\b", "S.A.", nome)
    return nome

def _estado_por_extenso(sigla_ou_nome: Optional[str]) -> Optional[str]:
    if not sigla_ou_nome:
        return None
    if len(sigla_ou_nome) == 2 and sigla_ou_nome.upper() in _UF_TO_ESTADO:
        return _UF_TO_ESTADO[sigla_ou_nome.upper()]
    alvo = sigla_ou_nome.strip()
    for _, nome in _UF_TO_ESTADO.items():
        if alvo.lower() == nome.lower():
            return nome
    return None

def extrair_campos(pdf_bytes: bytes) -> Dict[str, Optional[str]]:
    """
    Entrada: bytes do PDF
    Saída: {"nome_empresa": <str ou None>, "estado": <str ou None>}
    """
    texto = _texto_pdf(pdf_bytes)
    if not texto.strip():
        raise ValueError("Não foi possível extrair texto do PDF (arquivo vazio ou somente imagem).")
    nome_raw = _parse_nome_empresarial(texto)
    uf = _parse_uf(texto)
    nome_norm = _normalizar_nome_empresa(nome_raw or "")
    estado = _estado_por_extenso(uf)
    return {"nome_empresa": nome_norm or None, "estado": estado or None}