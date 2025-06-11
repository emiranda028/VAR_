import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from PIL import Image

st.set_page_config(layout="wide", page_title="VARGENTO - Análisis VAR Inteligente", page_icon="⚽")

@st.cache_resource
def cargar_y_entrenar():
    df = pd.read_csv("VAR_Limpio_Generado.csv")
    vectorizador = CountVectorizer()
    X = vectorizador.fit_transform(df["descripcion"])
    le = LabelEncoder()
    y = le.fit_transform(df["Decision"])
    modelo = XGBClassifier(n_estimators=10, max_depth=3, use_label_encoder=False, eval_metric="mlogloss")
    modelo.fit(X, y)
    return modelo, vectorizador, le

modelo, vectorizador, le = cargar_y_entrenar()

# Diseño
st.markdown("""
    <style>
        .title { font-size: 36px; font-weight: bold; color: #003366; }
        .subtitle { font-size: 20px; color: #333333; margin-bottom: 15px; }
        .footer { font-size: 13px; color: gray; margin-top: 40px; text-align: center; }
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    </style>
""", unsafe_allow_html=True)

st.image("https://media.tenor.com/xOb4uwv-VV8AAAAC/var-checking.gif", use_container_width=True)
st.markdown("<div class='title'>⚽ Bienvenido a VARGENTO</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>La plataforma inteligente para asistir decisiones arbitrales mediante IA</div>", unsafe_allow_html=True)

st.markdown("""
👉 Subí una imagen, video o link de YouTube de la jugada.  
👉 Describí brevemente lo ocurrido.  
👉 Recibí la sugerencia de decisión basada en el historial VAR.

📖 [Ver Reglamento de Juego FIFA](https://digitalhub.fifa.com/m/799749e5f64c0f86/original/lnc9zjo8xf2j3nvwfazh-pdf.pdf)
""", unsafe_allow_html=True)

st.markdown("---")
st.subheader("📸 Analizar nueva jugada")

texto_jugada = st.text_area("✍️ Describí la jugada:", "Jugador comete falta dentro del área tras revisión del VAR")
archivo_subido = st.file_uploader("📁 Subí una imagen o video de la jugada (opcional):", type=["jpg", "jpeg", "png", "mp4"])
link_youtube = st.text_input("🔗 O pegá un link de YouTube con la jugada (opcional):")

if st.button("🔍 Predecir decisión"):
    if not texto_jugada.strip():
        st.warning("Por favor ingresá una descripción válida.")
    else:
        X_nueva = vectorizador.transform([texto_jugada])
        pred = modelo.predict(X_nueva)
        pred_texto = le.inverse_transform(pred)[0]
        st.success(f"📢 Decisión sugerida: **{pred_texto}**")

        if archivo_subido:
            if archivo_subido.type.startswith("video"):
                st.video(archivo_subido)
            elif archivo_subido.type.startswith("image"):
                img = Image.open(archivo_subido)
                st.image(img, caption="📷 Imagen de la jugada")

        if link_youtube:
            st.video(link_youtube)

st.markdown("---")
st.markdown("<div class='footer'>Desarrollado por LTELC - Consultoría en Datos e IA ⚙️</div>", unsafe_allow_html=True)
