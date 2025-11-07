import streamlit as st
import os
import fitz  # PyMuPDF
import pandas as pd

# Configurações
PASTA_DESTINO = "/workspaces/scriptpy/PDF"
os.makedirs(PASTA_DESTINO, exist_ok=True)

st.title("📄 Upload e Extração de Dados de PDF")

# Upload do arquivo
arquivo = st.file_uploader("Envie um arquivo PDF", type=["pdf"])

if arquivo is not None:
    caminho_pdf = os.path.join(PASTA_DESTINO, arquivo.name)

    # Salvar o arquivo
    with open(caminho_pdf, "wb") as f:
        f.write(arquivo.getbuffer())
    st.success(f"Arquivo salvo em: {caminho_pdf}")

    # Parâmetros de extração
    pagina_inicial = st.number_input("Página inicial para leitura", min_value=1, value=6)
    colunas = ["Item", "CPF", "Nome", "Titulação", "Função", "Total de Horas", "Dedicação", "Valor"]
    dados_extraidos = []

    # Processar PDF
    doc = fitz.open(caminho_pdf)
    for page_num in range(pagina_inicial - 1, len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("dict")["blocks"]
        valores = {}

        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    texto = span["text"].strip()
                    if texto in colunas:
                        x_ref, y_ref = span["bbox"][0], span["bbox"][3]
                        valor_encontrado = ""
                        for b in blocks:
                            for l in b.get("lines", []):
                                for s in l.get("spans", []):
                                    x, y = s["bbox"][0], s["bbox"][1]
                                    if abs(x - x_ref) < 10 and y > y_ref:
                                        valor_encontrado = s["text"].strip()
                                        break
                                if valor_encontrado:
                                    break
                            if valor_encontrado:
                                break
                        valores[texto] = valor_encontrado

        if len(valores) == len(colunas):
            dados_extraidos.append(valores)

    # Gerar planilha
    if dados_extraidos:
        df = pd.DataFrame(dados_extraidos, columns=colunas)
        caminho_excel = os.path.join(PASTA_DESTINO, "dados_extraidos.xlsx")
        df.to_excel(caminho_excel, index=False)
        st.success("✅ Extração concluída!")
        st.download_button("📥 Baixar Excel", data=open(caminho_excel, "rb").read(), file_name="dados_extraidos.xlsx")
    else:
        st.warning("Nenhum dado foi extraído. Verifique se o layout do PDF está correto.")