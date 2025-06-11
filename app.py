import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

st.set_page_config(layout="wide", page_title="VARGENTO - Análisis VAR Inteligente", page_icon="⚽")

@st.cache_resource
def cargar_modelo():
    with open("modelo.pkl", "rb") as f:
        modelo = pickle.load(f)
    with open("vectorizador.pkl", "rb") as f:
        vectorizador = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return modelo, vectorizador, le

modelo, vectorizador, le = cargar_modelo()

st.title("⚽ VARGENTO - Análisis VAR Inteligente")
st.markdown("La plataforma inteligente para asistir en decisiones arbitrales mediante IA.")

texto = st.text_area("📋 Describí la jugada", "Falta dentro del área tras revisión del VAR")

if st.button("🔍 Predecir decisión"):
    if texto.strip():
        X_nueva = vectorizador.transform([texto])
        pred = modelo.predict(X_nueva)
        pred_label = le.inverse_transform(pred)[0]
        st.success(f"📢 Decisión sugerida: **{pred_label}**")
    else:
        st.warning("Ingresá una descripción válida.")
