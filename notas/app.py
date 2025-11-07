# app.py
# Abas:
#  1) Serviços Terceiros (Fornecedor + Nº Doc.)  -> salva servicos_df na sessão
#  2) Identificar Nº NF-e/NFS-e (PDF único ou ZIP) -> salva pdf_nf_bytes ou zip_nf_bytes na sessão
#  3) Renomear PDFs (Action) -> usa o que já está na sessão; não pede novo upload

import io
import sys
import zipfile
from pathlib import Path

import streamlit as st
import pandas as pd

# ---- Imports locais (notas/modulos) ----
ROOT = Path(__file__).resolve().parent
MOD_PATH = ROOT / "modulos"
if str(MOD_PATH) not in sys.path:
    sys.path.insert(0, str(MOD_PATH))

# Aba 1
parse_pdf_servicos, ERR_SERVICOS = None, None
try:
    from extrator_nf import parse_pdf as parse_pdf_servicos
except Exception as e:
    ERR_SERVICOS = e

# Aba 2
identify_extrair_nf, ERR_IDENTIFY = None, None
try:
    from identify import extrair_numero_nf as identify_extrair_nf
except Exception as e:
    ERR_IDENTIFY = e

# Aba 3 (Action)
action_process_zip_rename, ERR_ACTION = None, None
try:
    from action import process_zip_rename as action_process_zip_rename
except Exception as e:
    ERR_ACTION = e

# ---- UI principal ----
st.set_page_config(page_title="Notas • Extratores", layout="wide")
st.title("Notas • Extratores")

aba1, aba2, aba3 = st.tabs([
    "Serviços Terceiros (Fornecedor + Nº Doc.)",
    "Identificar Nº NF-e/NFS-e (PDF ou ZIP)",
    "Renomear PDFs (balancete × ZIP) — Action"
])

# ====================== ABA 1: Serviços Terceiros ======================
with aba1:
    st.subheader("Serviços Terceiros → Fornecedor + Nº Doc.")
    if parse_pdf_servicos is None:
        st.error(
            "Não foi possível importar `parse_pdf` de `extrator_nf.py`.\n\n"
            f"Erro: {ERR_SERVICOS}"
        )
    else:
        up1 = st.file_uploader("PDF do relatório (páginas ~8–16)", type=["pdf"], key="pdf_st")
        c1, c2, c3 = st.columns(3)
        with c1:
            p_ini = st.number_input("Página inicial", min_value=1, value=8, step=1, key="pini_st")
        with c2:
            p_fim = st.number_input("Página final", min_value=1, value=16, step=1, key="pfim_st")
        with c3:
            mode = st.selectbox("Modo de leitura", ["text", "blocks"], index=0, key="mode_st")

        if st.button("Extrair (Serviços Terceiros)", type="primary", use_container_width=True, key="btn_st"):
            if not up1:
                st.warning("Envie um PDF.")
                st.stop()

            pdf_bytes = up1.read()
            with st.spinner("Processando…"):
                try:
                    rows = parse_pdf_servicos(pdf_bytes=pdf_bytes, p_ini=int(p_ini), p_fim=int(p_fim), mode=mode)
                except Exception as e:
                    st.error(f"Falha ao processar o PDF: {e}")
                    st.stop()

            if not rows:
                st.error("Nenhum registro encontrado. Tente trocar o modo ('text' ↔ 'blocks') ou ajustar o intervalo de páginas.")
            else:
                seen, out = set(), []
                for r in rows:
                    key = (r.get("fornecedor"), r.get("cnpj_fornecedor"), r.get("numero_doc"), r.get("pagina"))
                    if key not in seen:
                        seen.add(key); out.append(r)

                df = pd.DataFrame(out, columns=["fornecedor", "cnpj_fornecedor", "numero_doc", "pagina"])
                st.success(f"Registros encontrados: {len(rows)} | Únicos: {len(df)}")
                st.dataframe(df, use_container_width=True, height=420)

                # Guarda balancete na sessão para a Action (Aba 3)
                st.session_state["servicos_df"] = df[["fornecedor", "cnpj_fornecedor", "numero_doc"]].copy()

                # Download CSV (opcional)
                csv_buf = io.StringIO()
                df.to_csv(csv_buf, index=False)
                st.download_button(
                    label="Baixar CSV (fornecedor_numdoc.csv)",
                    data=csv_buf.getvalue().encode("utf-8"),
                    file_name="fornecedor_numdoc.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="dl_st_csv"
                )

# ====================== ABA 2: Identify Nº NF (PDF único ou ZIP) ======================
with aba2:
    st.subheader("Identificar Nº da NF (NF-e / NFS-e)")
    if identify_extrair_nf is None:
        st.error(
            "Não foi possível importar `extrair_numero_nf` de `identify.py`.\n\n"
            f"Erro: {ERR_IDENTIFY}"
        )
    else:
        input_mode = st.radio(
            "Escolha a fonte:",
            options=["PDF único", "ZIP com vários PDFs"],
            horizontal=True,
            key="nf_src_mode"
        )
        max_pages = st.number_input("Máx. páginas a varrer por arquivo", min_value=1, value=3, step=1, key="maxpg_nf")

        if input_mode == "PDF único":
            up_pdf = st.file_uploader("Selecione um PDF", type=["pdf"], key="pdf_nf")
            if st.button("Identificar Nº da NF (PDF)", type="primary", use_container_width=True, key="btn_nf_pdf"):
                if not up_pdf:
                    st.warning("Envie um PDF.")
                    st.stop()
                pdf_bytes = up_pdf.read()

                # Guarda o PDF único na sessão (para Action usar depois)
                st.session_state["pdf_nf_bytes"] = pdf_bytes
                st.session_state["pdf_nf_name"] = up_pdf.name
                # Se havia ZIP salvo, limpamos (para não confundir a Action)
                st.session_state.pop("zip_nf_bytes", None)

                with st.spinner("Identificando…"):
                    try:
                        df_nf = identify_extrair_nf(pdf_bytes, max_pages=int(max_pages))
                    except Exception as e:
                        st.error(f"Falha ao processar: {e}")
                        st.stop()

                if df_nf.empty:
                    st.warning("Não foi possível identificar o número da NF nas páginas analisadas.")
                else:
                    df_nf.insert(0, "Arquivo", up_pdf.name)
                    st.success(f"Registros encontrados: {len(df_nf)}")
                    st.dataframe(df_nf, use_container_width=True, height=420)

                    buf = io.StringIO()
                    df_nf.to_csv(buf, index=False)
                    st.download_button(
                        label="Baixar CSV (numeros_nf.csv)",
                        data=buf.getvalue().encode("utf-8"),
                        file_name="numeros_nf.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="dl_nf_pdf_csv"
                    )

        else:
            up_zip = st.file_uploader("Selecione um ZIP contendo PDFs", type=["zip"], key="zip_nf")
            if st.button("Identificar Nº da NF (ZIP)", type="primary", use_container_width=True, key="btn_nf_zip"):
                if not up_zip:
                    st.warning("Envie um arquivo ZIP.")
                    st.stop()

                # Lê e guarda o ZIP na sessão (para Action usar depois)
                zip_bytes = up_zip.read()
                st.session_state["zip_nf_bytes"] = zip_bytes
                # Se havia PDF único salvo, limpamos
                st.session_state.pop("pdf_nf_bytes", None)
                st.session_state.pop("pdf_nf_name", None)

                all_rows = []
                errors = []
                with st.spinner("Processando ZIP…"):
                    try:
                        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
                            pdf_infos = [i for i in z.infolist() if (not i.is_dir()) and i.filename.lower().endswith(".pdf")]
                            if not pdf_infos:
                                st.warning("ZIP não contém PDFs.")
                            else:
                                prog = st.progress(0.0, text="Processando PDFs do ZIP…")
                                total = len(pdf_infos)
                                for idx, info in enumerate(pdf_infos, start=1):
                                    try:
                                        pdf_bytes_file = z.read(info.filename)
                                        df_nf = identify_extrair_nf(pdf_bytes_file, max_pages=int(max_pages))
                                        if not df_nf.empty:
                                            df_nf.insert(0, "Arquivo", info.filename)
                                            all_rows.append(df_nf)
                                    except Exception as e:
                                        errors.append((info.filename, str(e)))
                                    finally:
                                        prog.progress(min(idx/total, 1.0), text=f"{idx}/{total} arquivos")
                                prog.empty()
                    except zipfile.BadZipFile:
                        st.error("Arquivo ZIP inválido/corrompido.")
                        st.stop()

                df_all = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame(
                    columns=["Arquivo", "Página", "Número", "Origem", "Modo", "Trecho"]
                )
                st.success(f"Arquivos com resultado: {len(all_rows)} • Erros: {len(errors)}")
                st.dataframe(df_all, use_container_width=True, height=420)

                buf_zip = io.StringIO()
                df_all.to_csv(buf_zip, index=False)
                st.download_button(
                    label="Baixar CSV consolidado (numeros_nf_zip.csv)",
                    data=buf_zip.getvalue().encode("utf-8"),
                    file_name="numeros_nf_zip.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="dl_nf_zip_csv"
                )
                if errors:
                    with st.expander("Erros por arquivo"):
                        err_df = pd.DataFrame(errors, columns=["Arquivo", "Erro"])
                        st.dataframe(err_df, use_container_width=True, height=240)

# ====================== ABA 3: Action (somente front-end) ======================
with aba3:
    st.subheader("Renomear PDFs (balancete × ZIP) — usando uploads já feitos nas Abas 1 e 2")
    if action_process_zip_rename is None:
        st.error(
            "Não foi possível importar `process_zip_rename` de `action.py`.\n\n"
            f"Erro: {ERR_ACTION}"
        )
    else:
        # Mostrar status do que já temos na sessão
        bal_df = st.session_state.get("servicos_df")
        zip_bytes = st.session_state.get("zip_nf_bytes")
        pdf_bytes_single = st.session_state.get("pdf_nf_bytes")
        pdf_name_single = st.session_state.get("pdf_nf_name", "arquivo.pdf")

        colA, colB, colC = st.columns(3)
        with colA:
            st.metric("Balancete carregado (Aba 1)", "SIM" if (isinstance(bal_df, pd.DataFrame) and not bal_df.empty) else "NÃO")
        with colB:
            st.metric("ZIP carregado (Aba 2)", "SIM" if zip_bytes else "NÃO")
        with colC:
            st.metric("PDF único carregado (Aba 2)", "SIM" if pdf_bytes_single else "NÃO")

        st.caption("Dica: se você processou PDF único na Aba 2, a Action cria um ZIP temporário com esse arquivo.")

        if st.button("Processar e Renomear (sem novo upload)", type="primary", use_container_width=True, key="btn_action_run"):
            # Validações mínimas
            if bal_df is None or bal_df.empty:
                st.warning("Aba 1 ainda não foi executada ou não há balancete na sessão.")
                st.stop()
            # Construir o ZIP de entrada a partir da sessão
            if zip_bytes:
                zip_to_process = zip_bytes
            elif pdf_bytes_single:
                # Monta um ZIP temporário com o PDF único
                buf_tmp = io.BytesIO()
                with zipfile.ZipFile(buf_tmp, "w", compression=zipfile.ZIP_DEFLATED) as zout:
                    zout.writestr(pdf_name_single if pdf_name_single.lower().endswith(".pdf") else f"{pdf_name_single}.pdf", pdf_bytes_single)
                buf_tmp.seek(0)
                zip_to_process = buf_tmp.getvalue()
            else:
                st.warning("Aba 2 ainda não foi executada (nenhum PDF/ZIP salvo na sessão).")
                st.stop()

            with st.spinner("Renomeando…"):
                try:
                    # Defaults do fluxo (sem UI extra aqui)
                    new_zip_bytes, df_log = action_process_zip_rename(
                        zip_bytes=zip_to_process,
                        balancete_rows=bal_df.to_dict(orient="records"),
                        max_pages_per_pdf=3,
                        keep_unmatched=True,
                    )
                except Exception as e:
                    st.error(f"Falha no processo: {e}")
                    st.stop()

            st.success(f"Concluído. Linhas de log: {len(df_log)}")
            st.dataframe(df_log, use_container_width=True, height=420)

            # Download ZIP renomeado
            st.download_button(
                "Baixar ZIP renomeado",
                data=new_zip_bytes,
                file_name="renomeadas.zip",
                mime="application/zip",
                use_container_width=True,
                key="dl_zip_ren"
            )

            # Download LOG
            buf_log = io.StringIO(); df_log.to_csv(buf_log, index=False)
            st.download_button(
                "Baixar LOG (CSV)",
                data=buf_log.getvalue().encode("utf-8"),
                file_name="log_renomeacao.csv",
                mime="text/csv",
                use_container_width=True,
                key="dl_log_csv"
            )
