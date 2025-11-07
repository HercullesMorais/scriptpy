# appRH.py
# -*- coding: utf-8 -*-
import streamlit as st
import os
import re
import shutil
import pandas as pd
import fitz  # PyMuPDF
import unicodedata

# ========================
# Configuração inicial
# ========================
st.set_page_config(page_title="Extração — Relação de RH", layout="wide")
st.title("📄 Extração — Relação de Recursos Humanos")

# Pasta para saídas
if "sess_dir" not in st.session_state:
    os.makedirs("PDF", exist_ok=True)
    st.session_state["sess_dir"] = os.path.join("PDF", "sess_appRH")
    os.makedirs(st.session_state["sess_dir"], exist_ok=True)
PASTA_DESTINO = st.session_state["sess_dir"]

# Controle de reset do uploader
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0
if "caminho_pdf" not in st.session_state:
    st.session_state["caminho_pdf"] = None

# ========================
# UI principal
# ========================
arquivo = st.file_uploader(
    "Envie o arquivo PDF",
    type=["pdf"],
    key=f"uploader_{st.session_state['uploader_key']}"
)
pagina_inicial = st.number_input("Página inicial da seção", min_value=1, value=3, step=1)

modo_layout = st.radio(
    "Identificação do layout",
    ["Auto (recomendado)", "Modelo 1 (atual)", "Modelo 2 (novo)"],
    index=0, horizontal=True
)

mostrar_debug = st.checkbox("🔎 Mostrar debug (texto/linhas e matches)", value=False)
parar_no_total_m1 = st.checkbox("🛑 (Modelo 1) Parar quando encontrar 'Total R$'", value=True)

# ========================
# Padrões (Modelo 1)
# ========================
CPF_REGEX_MASK = r"(?:\d{3}\.){2}\d{3}\-\d{2}"
TITULACOES_SET = [
    r"Graduado", r"P[óo]s-?Graduado", r"Especialista", r"Mestre", r"Doutor",
    r"Tecn[óo]logo", r"Ensino\s+M[ée]dio", r"Superior\s+Completo", r"Superior\s+Incompleto",
    r"Bacharel"
]
DEDIC_SET = [
    r"Parcial", r"Integral", r"Exclusiva", r"Tempo\s+Parcial", r"Tempo\s+Integral"
]
TITULACOES = r"(" + r"|".join(TITULACOES_SET) + r")"
DEDIC_REGEX = r"(" + r"|".join(DEDIC_SET) + r")"

PADRAO_REGISTRO_M1 = re.compile(
    rf"""
    (?P<Item>\d+)\s+
    (?P<CPF>{CPF_REGEX_MASK})\s+
    (?P<Nome>.+?)\s+
    (?P<Titulacao>{TITULACOES})\s+
    (?P<Funcao>.+?)\s+
    (?P<Horas>\d{{1,4}})\s+
    (?P<Dedicacao>{DEDIC_REGEX})\s+
    (?P<Valor>R\$\s*[\d\.,]+)
    """,
    re.IGNORECASE | re.VERBOSE
)
PADRAO_TOTAL_M1 = re.compile(r"Total\s*R\$\s*[\d\.,]*", flags=re.IGNORECASE)

# ========================
# Padrões/rotinas (Modelo 2)
# ========================
HEADER_REGEX = re.compile(
    r"3\.1\.16\.2\.8\.\s*RELAÇÃO\s+DE\s+RECURSOS\s+HUMANOS",
    re.IGNORECASE
)
TOTAL_REGEX = re.compile(
    r"^\s*TOTAL(?:\s+GERAL)?\s*(?:R\$\s*[\d\.,]+)?\s*$",
    re.IGNORECASE | re.MULTILINE
)
RE_ITEM  = re.compile(r"^\s*Item[:\s]+(\d+)\s*$", re.IGNORECASE)
RE_CPF   = re.compile(r"^\s*CPF[:\s]+(\d{11}|\d{3}\.\d{3}\.\d{3}\-\d{2})\s*$", re.IGNORECASE)
RE_NOME  = re.compile(r"^\s*Nome[:\s]+(.+)$", re.IGNORECASE)
RE_TITU  = re.compile(r"^\s*Titula[cç][aã]o[:\s]+(.+)$", re.IGNORECASE)
RE_HORAS = re.compile(r"^\s*Total(?:\s+de)?\s+Horas\s*\(Anual\)[:\s]+([\d\.,]+)\s*$", re.IGNORECASE)
RE_DEDIC = re.compile(r"^\s*Dedica[cç][aã]o[:\s]+(.+)$", re.IGNORECASE)
RE_VALOR = re.compile(r"^\s*Valor\s*\(R\$\)[:\s]*(?:R\$\s*)?([\d\.,]+)\s*$", re.IGNORECASE)

# ========================
# Utilitários
# ========================
def normaliza_espacos(t: str) -> str:
    t = (t or "").replace("\xa0", " ")
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s*/\s*", "/", t)
    return t.strip()

def normaliza_funcao(txt: str) -> str:
    txt = re.sub(r'(?<!Nível\s)(?<!Classe\s)(?<!Faixa\s)\b\d+\b', ' ', f' {txt or ""} ').strip()
    txt = re.sub(r"\bSN\b|\bSn\b", "Sênior", txt, flags=re.IGNORECASE)
    txt = re.sub(r"\bJr\b", "Júnior", txt)
    txt = re.sub(r"\bPl\b|\bPL\b", "Pleno", txt)
    return re.sub(r"\s{2,}", " ", txt).strip()

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

def valida_cpf(cpf: str) -> bool:
    nums = re.sub(r'\D', '', cpf or '')
    if len(nums) != 11 or nums == nums[0]*11:
        return False
    def dv(digs):
        s = sum(int(d)*w for d, w in zip(digs, range(len(digs)+1, 1, -1)))
        r = (s * 10) % 11
        return '0' if r == 10 else str(r)
    return nums[9] == dv(nums[:9]) and nums[10] == dv(nums[:10])

def formata_cpf(cpf: str) -> str:
    nums = re.sub(r'\D', '', cpf or '')
    if len(nums) == 11:
        return f"{nums[0:3]}.{nums[3:6]}.{nums[6:9]}-{nums[9:11]}"
    return cpf or ""

def to_ascii_upper(s: str) -> str:
    s_norm = unicodedata.normalize("NFKD", s or "")
    s_noacc = "".join(c for c in s_norm if not unicodedata.combining(c))
    return s_noacc.upper()

def norm_text_keep_lines(s: str) -> str:
    s = (s or "").replace("\xa0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    return s

# ========================
# Extratores
# ========================
def extrai_m1_por_pagina(texto_raw: str, aplicar_recorte_total: bool) -> list[dict]:
    texto_norm = normaliza_espacos(texto_raw or "")
    if aplicar_recorte_total:
        cab = "RELAÇÃO DE RECURSOS HUMANOS"
        idx_cab = texto_norm.upper().find(cab)
        if idx_cab != -1:
            texto_norm = texto_norm[idx_cab:]
        m_total = PADRAO_TOTAL_M1.search(texto_norm)
        if m_total:
            texto_norm = texto_norm[:m_total.start()].strip()
    resultados = []
    for m in PADRAO_REGISTRO_M1.finditer(texto_norm):
        r = {
            "Item": m.group("Item"),
            "CPF": m.group("CPF"),
            "Nome": (m.group("Nome") or "").strip(),
            "Titulação": (m.group("Titulacao") or "").strip(),
            "Função": normaliza_funcao((m.group("Funcao") or "").strip()),
            "Total de Horas": (m.group("Horas") or "").strip(),
            "Dedicação": (m.group("Dedicacao") or "").strip(),
            "Valor R$": (m.group("Valor") or "").strip(),
            "Valor (num)": valor_para_float(m.group("Valor")),
        }
        resultados.append(r)
    return resultados

def extrair_secao_itens_m2(pdf_path: str, start_page: int):
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
    atual = None

    def push_atual():
        nonlocal atual
        if not atual:
            return
        if atual.get("Item") and (atual.get("CPF") or atual.get("Nome")):
            if atual.get("Valor R$"):
                atual["Valor (num)"] = valor_para_float(atual["Valor R$"])
            else:
                atual["Valor (num)"] = None
            if atual.get("CPF"):
                atual["CPF"] = formata_cpf(atual["CPF"])
            if atual.get("Total de Horas") is not None:
                h = horas_para_int(str(atual["Total de Horas"]))
                if h is not None:
                    atual["Total de Horas"] = h
            atual.setdefault("Titulação", "")
            atual.setdefault("Função", "")
            atual.setdefault("Dedicação", "")
            registros.append(atual.copy())
        atual = None

    i = 0
    n = len(linhas)
    while i < n:
        ln = linhas[i]
        nxt = linhas[i+1] if i+1 < n else ""
        comb = (ln + " " + nxt).strip()

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

        if not atual["CPF"]:
            m = RE_CPF.match(ln) or RE_CPF.match(comb)
            if m:
                atual["CPF"] = m.group(1)
                i += 2 if (not RE_CPF.match(ln) and nxt) else 1
                continue

        if not atual["Nome"]:
            m = RE_NOME.match(ln) or RE_NOME.match(comb)
            if m:
                atual["Nome"] = m.group(1).strip()
                i += 2 if (not RE_NOME.match(ln) and nxt) else 1
                continue

        if not atual["Titulação"]:
            m = RE_TITU.match(ln) or RE_TITU.match(comb)
            if m:
                atual["Titulação"] = m.group(1).strip()
                i += 2 if (not RE_TITU.match(ln) and nxt) else 1
                continue

        if atual["Total de Horas"] is None:
            m = RE_HORAS.match(ln) or RE_HORAS.match(comb)
            if m:
                atual["Total de Horas"] = m.group(1)
                i += 2 if (not RE_HORAS.match(ln) and nxt) else 1
                continue

        if not atual["Dedicação"]:
            m = RE_DEDIC.match(ln) or RE_DEDIC.match(comb)
            if m:
                atual["Dedicação"] = m.group(1).strip()
                i += 2 if (not RE_DEDIC.match(ln) and nxt) else 1
                continue

        if atual["Valor R$"] is None:
            m = RE_VALOR.match(ln) or RE_VALOR.match(comb)
            if m:
                valor_str = m.group(1)
                atual["Valor R$"] = f"R$ {valor_str}"
                push_atual()
                i += 2 if (not RE_VALOR.match(ln) and nxt) else 1
                continue

        i += 1

    push_atual()
    return registros, aviso, total_encontrado

# ========================
# Reset
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
# Execução
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
        df_final = None
        layout_escolhido = None
        aviso_m2 = None

        if modo_layout == "Modelo 1 (atual)":
            registros = []
            with fitz.open(caminho_pdf) as doc:
                for i in range(pagina_inicial - 1, len(doc)):
                    texto_raw = doc.load_page(i).get_text("text") or ""
                    encontrados = extrai_m1_por_pagina(texto_raw, aplicar_recorte_total=parar_no_total_m1)
                    for r in encontrados:
                        r["Layout"] = "Modelo 1"
                    registros.extend(encontrados)
                    if parar_no_total_m1 and PADRAO_TOTAL_M1.search(normaliza_espacos(texto_raw)):
                        break
            if registros:
                df_final = pd.DataFrame(registros)
                layout_escolhido = "Modelo 1"

        elif modo_layout == "Modelo 2 (novo)":
            registros, aviso_m2, _ = extrair_secao_itens_m2(caminho_pdf, start_page=int(pagina_inicial))
            for r in registros:
                r["Layout"] = "Modelo 2"
            if registros:
                df_final = pd.DataFrame(registros)
                layout_escolhido = "Modelo 2"

        else:  # Auto (recomendado)
            registros_m1 = []
            with fitz.open(caminho_pdf) as doc:
                for i in range(pagina_inicial - 1, len(doc)):
                    texto_raw = doc.load_page(i).get_text("text") or ""
                    encontrados = extrai_m1_por_pagina(texto_raw, aplicar_recorte_total=parar_no_total_m1)
                    for r in encontrados:
                        r["Layout"] = "Modelo 1"
                    registros_m1.extend(encontrados)
                    if parar_no_total_m1 and PADRAO_TOTAL_M1.search(normaliza_espacos(texto_raw)):
                        break

            registros_m2, aviso_m2, _ = extrair_secao_itens_m2(caminho_pdf, start_page=int(pagina_inicial))
            for r in registros_m2:
                r["Layout"] = "Modelo 2"

            # -------- CORREÇÃO AQUI --------
            def score(rows):
                """Pontua a lista de registros: +1 para cada registro com CPF válido e algum Valor."""
                if not rows:
                    return 0
                pts = 0
                for r in rows:
                    has_val = (r.get("Valor (num)") is not None) or bool(r.get("Valor R$"))
                    if valida_cpf(r.get("CPF", "")) and has_val:
                        pts += 1
                return pts

            s1 = score(registros_m1)
            s2 = score(registros_m2)
            if s2 > s1 or (s2 == s1 and len(registros_m2) > len(registros_m1)):
                if registros_m2:
                    df_final = pd.DataFrame(registros_m2)
                    layout_escolhido = "Modelo 2"
                elif registros_m1:
                    df_final = pd.DataFrame(registros_m1)
                    layout_escolhido = "Modelo 1"
            else:
                if registros_m1:
                    df_final = pd.DataFrame(registros_m1)
                    layout_escolhido = "Modelo 1"
                elif registros_m2:
                    df_final = pd.DataFrame(registros_m2)
                    layout_escolhido = "Modelo 2"

        if df_final is not None and not df_final.empty:
            if "Item" in df_final.columns:
                df_final["Item"] = pd.to_numeric(df_final["Item"], errors="coerce")
                df_final = df_final.sort_values(["Item"], kind="stable")
            if "Total de Horas" in df_final.columns:
                df_final["Total de Horas"] = pd.to_numeric(df_final["Total de Horas"], errors="coerce")

            if aviso_m2:
                st.info(aviso_m2)

            st.success(f"✅ Extraídos {len(df_final)} registro(s). Layout escolhido: **{layout_escolhido or '—'}**.")
            st.dataframe(df_final, use_container_width=True)

            caminho_excel = os.path.join(PASTA_DESTINO, "tabela_reconstruida.xlsx")
            caminho_csv = os.path.join(PASTA_DESTINO, "tabela_reconstruida.csv")
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
        else:
            st.error("Nenhum registro foi encontrado. Verifique a página inicial e o layout selecionado.")
            if st.button("🔄 Nova extração"):
                reset_app()
else:
    st.info("Envie um PDF e clique em **Extrair dados** para iniciar.")