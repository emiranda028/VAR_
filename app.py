import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Configurar página (esto debe ir primero)
st.set_page_config(page_title="Predicción VAR", layout="centered")

# Cargar y preparar datos
@st.cache_data
def cargar_datos():
    df = pd.read_csv("EPL_VAR_2000_jugadas.csv")
    df = df.dropna(subset=["descripcion", "decision"])
    return df

@st.cache_data
def entrenar_modelo(df):
    vectorizador = CountVectorizer()
    X = vectorizador.fit_transform(df["descripcion"])
    le = LabelEncoder()
    y = le.fit_transform(df["decision"])
    modelo = MultinomialNB()
    modelo.fit(X, y)
    return modelo, vectorizador, le

# UI principal
st.title("⚽ Predicción automática de decisiones VAR")
st.markdown("Subí una descripción textual de una jugada para que el sistema sugiera una decisión según el reglamento FIFA")

# Cargar datos y modelo
df = cargar_datos()
modelo, vectorizador, le = entrenar_modelo(df)

# Input del usuario
descripcion = st.text_area("📝 Describí la jugada con claridad")

# Predicción
if st.button("🔍 Predecir decisión"):
    if descripcion.strip() == "":
        st.warning("Por favor, ingresá una descripción válida.")
    else:
        descripcion_vectorizada = vectorizador.transform([descripcion])
        pred = modelo.predict(descripcion_vectorizada)
        pred_proba = modelo.predict_proba(descripcion_vectorizada)[0]
        decision = le.inverse_transform(pred)[0]
        confianza = np.max(pred_proba) * 100

        st.success(f"📢 Decisión sugerida: **{decision}** ({confianza:.2f}% confianza)")

        reglas = {
            "Penal": "Regla 12: Faltas y conducta incorrecta.",
            "No penal": "Regla 12: Contacto legal, no sancionable.",
            "Roja": "Regla 12: Conducta violenta o juego brusco grave.",
            "Amarilla": "Regla 12: Conducta antideportiva.",
            "Offside": "Regla 11: Posición adelantada.",
            "Gol válido": "Regla 10: El gol es válido si no hay infracciones.",
            "Gol anulado": "Regla 10 y 12: El gol se anula por infracción previa."
        }

        if decision in reglas:
            st.info(f"📘 Según el reglamento FIFA: {reglas[decision]}")

# Pie de página
st.markdown("---")
st.markdown('<div style="text-align: center; color: gray;">Desarrollado por LTELC - Consultoría en Datos e IA ⚙️</div>', unsafe_allow_html=True)
