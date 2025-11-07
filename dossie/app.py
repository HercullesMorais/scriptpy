# app.py
# -*- coding: utf-8 -*-
import re
import base64
import hashlib
from io import BytesIO
from typing import List, Tuple

import streamlit as st
import streamlit.components.v1 as components

from modulos.extrator import extrair_campos
from modulos.action import preencher_modelo

st.set_page_config(page_title="Extrator & Preenchimento — CNPJ/Word", page_icon="🧾", layout="centered")

st.title("🧾 Extrator CNPJ — Nome & Estado + 📝 Preenchimento Word")
st.caption("Faça upload do PDF, do modelo .docx e da LOGO (imagem) **no mesmo input**. Depois clique em **Processar** e **Gerar documento**.")

for key in ["pdf_bytes","docx_bytes","logo_bytes","pdf_name","docx_name","logo_name","resultado","docx_out","ano","fname","digest"]:
    if key not in st.session_state:
        st.session_state[key] = None

def _classificar(files) -> Tuple[bytes, str, bytes, str, bytes, str]:
    pdf_b = pdf_n = docx_b = docx_n = img_b = img_n = None, None, None, None, None, None
    pdf_b = docx_b = img_b = None
    pdf_n = docx_n = img_n = None
    for f in files:
        name = (f.name or "").lower()
        if name.endswith(".pdf") and pdf_b is None:
            pdf_b, pdf_n = f.getvalue(), f.name
        elif name.endswith(".docx") and docx_b is None:
            docx_b, docx_n = f.getvalue(), f.name
        elif any(name.endswith(ext) for ext in (".png",".jpg",".jpeg",".webp",".tiff",".bmp")) and img_b is None:
            img_b, img_n = f.getvalue(), f.name
    return pdf_b, pdf_n, docx_b, docx_n, img_b, img_n

st.subheader("Uploads (1 único input — selecione 3 arquivos)")
files = st.file_uploader(
    "Envie: 1) PDF do CNPJ  2) Modelo .docx  3) LOGO (imagem)",
    type=["pdf","docx","png","jpg","jpeg","webp","tiff","bmp"],
    accept_multiple_files=True
)

if files:
    pdf_b, pdf_n, docx_b, docx_n, img_b, img_n = _classificar(files)
    if pdf_b:
        st.session_state.pdf_bytes, st.session_state.pdf_name = pdf_b, pdf_n
    if docx_b:
        st.session_state.docx_bytes, st.session_state.docx_name = docx_b, docx_n
    if img_b:
        st.session_state.logo_bytes, st.session_state.logo_name = img_b, img_n

cols = st.columns(3)
with cols[0]:
    st.write("PDF:", f"**{st.session_state.pdf_name or '—'}**")
with cols[1]:
    st.write("DOCX:", f"**{st.session_state.docx_name or '—'}**")
with cols[2]:
    st.write("LOGO:", f"**{st.session_state.logo_name or '—'}**")

if st.session_state.logo_bytes:
    st.image(st.session_state.logo_bytes, caption=st.session_state.logo_name or "LOGO", use_container_width=True)

st.markdown("---")

st.session_state.ano = st.text_input("Insira o ano de candidatura antes de processar", value=st.session_state.ano or "", max_chars=4, placeholder="YYYY")

c1, c2, c3 = st.columns([1,1,1])
with c1:
    processar = st.button("⚙️ Processar", type="primary", use_container_width=True)
with c2:
    gerar = st.button("🧩 Gerar documento", use_container_width=True)
with c3:
    if st.button("🔄 Limpar", use_container_width=True):
        for k in list(st.session_state.keys()):
            st.session_state[k] = None
        st.rerun()

if processar:
    st.session_state.resultado = None
    if not st.session_state.pdf_bytes:
        st.warning("Envie o PDF do CNPJ.")
    else:
        try:
            st.session_state.resultado = extrair_campos(st.session_state.pdf_bytes)
            st.success("Extração concluída!")
            st.json(st.session_state.resultado, expanded=False)
        except Exception as e:
            st.error(f"Erro ao processar o PDF: {e}")

if gerar:
    st.session_state.docx_out = None
    if not st.session_state.resultado and st.session_state.pdf_bytes:
        try:
            st.session_state.resultado = extrair_campos(st.session_state.pdf_bytes)
        except Exception as e:
            st.error(f"Erro ao processar o PDF: {e}")
            st.stop()

    if not (st.session_state.ano and re.fullmatch(r"\d{4}", st.session_state.ano)):
        st.warning("Informe um ano válido (YYYY).")
        st.stop()
    if not st.session_state.docx_bytes:
        st.warning("Envie o modelo .docx.")
        st.stop()
    if not st.session_state.logo_bytes:
        st.warning("Envie a LOGO (imagem).")
        st.stop()

    nome = (st.session_state.resultado or {}).get("nome_empresa")
    estado = (st.session_state.resultado or {}).get("estado")
    if not nome or not estado:
        st.error("Não foi possível obter Nome/Estado do PDF.")
        st.stop()

    try:
        # Por enquanto: só deletamos a imagem da capa e trocamos textos.
        out = preencher_modelo(
            docx_bytes=st.session_state.docx_bytes,
            nome_empresa=nome,
            estado_extenso=estado,
            ano_candidatura=st.session_state.ano,
            logo_bytes=None,
        )
        if not isinstance(out, (bytes, bytearray)) or len(out) == 0:
            raise ValueError("Falha ao gerar .docx")
        st.session_state.docx_out = bytes(out)
        empresa_slug = re.sub(r"[^a-zA-Z0-9\-_]+", "", re.sub(r"\s+", "-", nome)).lower()[:60] or "documento"
        st.session_state.fname = f"modelo_preenchido-{st.session_state.ano}-{empresa_slug}.docx"
        st.success("Documento gerado com sucesso!")
    except Exception as e:
        st.error(f"Erro ao gerar o documento: {e}")

if st.session_state.docx_out:
    blob = st.session_state.docx_out
    fname = st.session_state.fname or "modelo_preenchido.docx"
    digest = hashlib.sha256(blob).hexdigest()[:12]

    st.download_button(
        "⬇️ Baixar (.docx)",
        data=BytesIO(blob),
        file_name=fname,
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        key=f"dl_btn_{digest}",
        use_container_width=True,
    )

    b64 = base64.b64encode(blob).decode("utf-8")
    st.link_button(
        "🔗 Abrir em nova guia (alternativo)",
        url=f"data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,{b64}",
        use_container_width=True,
    )

    if st.button("🚀 Forçar download (fallback)", use_container_width=True):
        components.html(
            f"""
            <html><body>
            <a id="force_dl" download="{fname}"
               hrefprocessingml.document;base64,{b64}</a>
            <script>document.getElementById('force_dl').click();</script>
            </body></html>
            """,
            height=0,
        )