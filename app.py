import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Configuración de la página (debe ser lo primero)
st.set_page_config(page_title="VARGENTO ⚽️", layout="wide")

# Cargar modelo, vectorizador y encoder
@st.cache_resource
def cargar_componentes():
    modelo = joblib.load("modelo_var_nb.pkl")
    vectorizador = joblib.load("vectorizador.pkl")
    le = joblib.load("label_encoder.pkl")
    return modelo, vectorizador, le

modelo, vectorizador, le = cargar_componentes()

# Estilo y título principal
st.title("📺 VARGENTO - Sistema de asistencia arbitral")
st.markdown("Bienvenido a la demo del modelo de predicción de decisiones arbitrales en jugadas de fútbol. ⚽️")
st.markdown("👉 Subí una imagen, video o link de YouTube de la jugada. Luego describila en texto y obtené una decisión sugerida por el sistema.")

# Inputs multimedia
col1, col2 = st.columns(2)

with col1:
    imagen = st.file_uploader("📷 Subí una imagen de la jugada", type=["png", "jpg", "jpeg"])
    if imagen:
        st.image(imagen, caption="Jugada cargada", use_container_width=True)

with col2:
    video_link = st.text_input("📹 Link de YouTube de la jugada")
    if video_link:
        st.video(video_link)

# Input de texto descriptivo
descripcion = st.text_area("📝 Descripción de la jugada:", placeholder="Ejemplo: Mano dentro del área tras rebote")

if st.button("🔍 Predecir decisión"):
    if descripcion.strip() == "":
        st.warning("Por favor, ingresá una descripción válida de la jugada.")
    else:
        descripcion_vectorizada = vectorizador.transform([descripcion])
        pred_proba = modelo.predict_proba(descripcion_vectorizada)[0]
        pred_idx = np.argmax(pred_proba)
        decision = le.inverse_transform([pred_idx])[0]
        confianza = pred_proba[pred_idx] * 100

        st.success(f"🧠 Decisión sugerida: **{decision}** ({confianza:.2f}% de confianza)")

        # Buscar referencia al reglamento FIFA según la decisión
        referencias = {
            "Penal": "Regla 12 - Faltas y conducta incorrecta (Infracciones dentro del área)",
            "Roja": "Regla 12 - Conducta violenta o juego brusco grave",
            "Amarilla": "Regla 12 - Conducta antideportiva",
            "Gol anulado": "Regla 11 - Fuera de juego o mano previa",
            "Gol válido": "Regla 10 - Gol marcado correctamente",
            "No penal": "Regla 12 - Contacto legal o sin infracción",
            "Offside": "Regla 11 - Posición de fuera de juego"
        }

        if decision in referencias:
            st.info(f"📘 Referencia al reglamento FIFA: {referencias[decision]}")

st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'>Desarrollado por LTELC - Consultoría en Datos e IA ⚙️</div>", unsafe_allow_html=True)

