import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import plotly.express as px
from fpdf import FPDF
import io
import base64
from PIL import Image

st.set_page_config(page_title="VARGENTO - Análisis VAR Inteligente", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("⚽ VARGENTO - Plataforma Inteligente de Análisis VAR")
st.markdown("Bienvenido a la plataforma de análisis de jugadas de fútbol con IA.")

@st.cache_resource
def cargar_y_entrenar():
    df = pd.read_csv("VAR_Limpio_Generado.csv")
    df = df.dropna(subset=["descripcion", "Decision"])
    df = df[df["descripcion"].str.strip() != ""]
    df = df[df["Decision"].str.strip() != ""]

    le = LabelEncoder()
    df["Decision_encoded"] = le.fit_transform(df["Decision"])

    vectorizador = CountVectorizer()
    X = vectorizador.fit_transform(df["descripcion"])
    y = df["Decision_encoded"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    modelo = MultinomialNB()
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.success(f"✅ Modelo entrenado con precisión: {acc*100:.2f}%")

    return modelo, vectorizador, le, df

modelo, vectorizador, le, df_filtrado = cargar_y_entrenar()

st.markdown("---")

st.header("📸 Subí una jugada para analizar")
descripcion = st.text_area("Describí la jugada:", "")
archivo = st.file_uploader("Subí una imagen (opcional):", type=["png", "jpg", "jpeg"])
video = st.file_uploader("Subí un video (opcional):", type=["mp4"])
link_youtube = st.text_input("O pegá un link de YouTube (opcional):")

if st.button("🔍 Predecir decisión"):
    if not descripcion.strip():
        st.warning("Por favor, ingresá una descripción.")
    else:
        X_nueva = vectorizador.transform([descripcion])
        probs = modelo.predict_proba(X_nueva)[0]
        prediccion_idx = probs.argmax()
        prediccion = le.inverse_transform([prediccion_idx])[0]
        probabilidad = probs[prediccion_idx] * 100

        st.success(f"📢 Decisión sugerida: **{prediccion}** ({probabilidad:.2f}% confianza)")

        st.subheader("🎞️ Contenido multimedia")
        if archivo:
            img = Image.open(archivo)
            st.image(img, caption="Imagen subida", use_container_width=True)
        if video:
            st.video(video)
        if link_youtube:
            st.video(link_youtube)

        st.subheader("📄 Exportar a PDF")
        if st.button("📥 Descargar reporte PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            texto_pdf = f"""Jugada: {descripcion}

Decisión sugerida: {prediccion}

Probabilidad: {probabilidad:.2f}%"""
            pdf.multi_cell(0, 10, texto_pdf)

            pdf_output = io.BytesIO()
            pdf.output(pdf_output)
            pdf_output.seek(0)
            b64 = base64.b64encode(pdf_output.read()).decode('utf-8')
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="reporte_var.pdf">📄 Descargar PDF</a>'
            st.markdown(href, unsafe_allow_html=True)

st.markdown("---")
st.subheader("📊 Visualización de decisiones en el dataset")
fig = px.histogram(df_filtrado, x="Decision", title="Distribución de decisiones")
st.plotly_chart(fig)

st.markdown('<div style="text-align: center; color: gray;">Desarrollado por LTELC - Consultoría en Datos e IA ⚙️</div>', unsafe_allow_html=True)
