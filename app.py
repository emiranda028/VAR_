
import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import io
from fpdf import FPDF
import base64

st.set_page_config(layout="wide", page_title="VARGENTO - Análisis VAR Inteligente", page_icon="⚽")

st.markdown("""
    <style>
        .title { font-size: 36px; font-weight: bold; color: #003366; }
        .subtitle { font-size: 20px; color: #333333; margin-bottom: 15px; }
        .footer { font-size: 13px; color: gray; margin-top: 40px; text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.image("https://media.tenor.com/xOb4uwv-VV8AAAAC/var-checking.gif", use_container_width=True)

st.markdown("# ⚽ Bienvenido a VARGENTO")
st.markdown("La plataforma inteligente para asistir en decisiones arbitrales mediante IA y análisis de jugadas.")
st.markdown("👉 Subí una imagen, video o link de YouTube de la jugada.  
👉 Describí brevemente lo ocurrido.  
👉 Recibí la sugerencia de decisión basada en el historial VAR.")
st.markdown("📖 [Ver Reglamento de Juego FIFA](https://digitalhub.fifa.com/m/799749e5f64c0f86/original/lnc9zjo8xf2j3nvwfazh-pdf.pdf)")
st.markdown("---")

@st.cache_resource
def cargar_modelo():
    modelo = joblib.load("modelo_var_nb.pkl")
    vectorizador = joblib.load("vectorizer_var.pkl")
    le = joblib.load("label_encoder_var.pkl")
    return modelo, vectorizador, le

modelo, vectorizador, le = cargar_modelo()

st.subheader("📸 Analizar nueva jugada")
descripcion = st.text_area("Describí la jugada:", "Jugador comete falta dentro del área tras revisión del VAR")
archivo_subido = st.file_uploader("Subí una imagen o video de la jugada (opcional):", type=["jpg", "jpeg", "png", "mp4"])
link_youtube = st.text_input("O pegá un link de YouTube con la jugada (opcional):")

if st.button("🔍 Predecir decisión"):
    if not descripcion.strip():
        st.warning("Por favor ingresá una descripción válida.")
    else:
        X_nueva = vectorizador.transform([descripcion])
        proba = modelo.predict_proba(X_nueva)[0]
        pred = modelo.predict(X_nueva)[0]
        decision = le.inverse_transform([pred])[0]
        confianza = proba[pred] * 100

        st.success(f"📢 Decisión sugerida: **{decision}** ({confianza:.2f}% confianza)")

        if archivo_subido:
            if archivo_subido.type.startswith("video"):
                st.video(archivo_subido)
            elif archivo_subido.type.startswith("image"):
                img = Image.open(archivo_subido)
                st.image(img, caption="Imagen de la jugada")

        if link_youtube:
            st.video(link_youtube)

        st.markdown("---")
        st.subheader("📥 Exportar a PDF")
        if st.button("📄 Descargar reporte"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            texto = f"Jugada: {descripcion}\n\nDecisión sugerida: {decision} ({confianza:.2f}% confianza)"
            for line in texto.split("\n"):
                pdf.multi_cell(0, 10, line)
            pdf_output = io.BytesIO()
            pdf.output(pdf_output)
            pdf_output.seek(0)
            b64 = base64.b64encode(pdf_output.read()).decode('utf-8')
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="reporte_var.pdf">📥 Descargar PDF</a>'
            st.markdown(href, unsafe_allow_html=True)

st.markdown("---")
st.markdown('<div class="footer">Desarrollado por LTELC - Consultoría en Datos e IA ⚙️</div>', unsafe_allow_html=True)
