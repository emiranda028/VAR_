import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import base64

st.set_page_config(page_title="VARGENTO - Análisis VAR", layout="centered")
st.title("⚽ VARgento: Asistente de Decisiones Arbitrales")

st.markdown("""
Bienvenido a **VARGENTO**, un sistema de predicción de decisiones arbitrales basado en inteligencia artificial.
Subí una jugada, describila y obtené:
- Una decisión sugerida
- El porcentaje de confianza
- La regla FIFA relacionada

🟢 *Buscamos reducir la discrecionalidad arbitral en el fútbol profesional.*
""")

st.markdown("---")

st.subheader("📸 Subí imagen, video o link de la jugada")
st.file_uploader("Imagen o video de la jugada", type=["jpg", "jpeg", "png", "mp4"])
link = st.text_input("Link de YouTube de la jugada")

# Cargar datos y entrenar modelo
@st.cache_data
def cargar_y_entrenar():
    df = pd.read_csv("jugadas_var_sinteticas.csv", encoding="utf-8-sig")
    df = df.dropna(subset=["descripcion", "decision"])

    vectorizador = CountVectorizer()
    X = vectorizador.fit_transform(df["descripcion"])
    y = df["decision"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo = MultinomialNB()
    modelo.fit(X_train, y_train)

    acc = accuracy_score(y_test, modelo.predict(X_test))
    return modelo, vectorizador, acc, df

modelo, vectorizador, acc, df_data = cargar_y_entrenar()

st.markdown(f"### 🎯 Precisión del modelo: {acc*100:.2f}%")

st.subheader("📝 Describí la jugada")
descripcion = st.text_area("Ejemplo: 'El defensor salta con el brazo extendido e impacta el balón dentro del área'.")

if st.button("📊 Predecir decisión"):
    if descripcion:
        X_nueva = vectorizador.transform([descripcion])
        pred = modelo.predict(X_nueva)[0]
        proba = max(modelo.predict_proba(X_nueva)[0]) * 100

        # Buscar regla relacionada
        fila = df_data[df_data["decision"] == pred].iloc[0]
        regla = fila["regla_fifa"]

        st.success(f"✅ Decisión sugerida: **{pred}**")
        st.info(f"📈 Confianza del modelo: {proba:.2f}%")
        st.markdown(f"📘 **Reglamento FIFA relacionado:** {regla}")
    else:
        st.warning("Por favor, escribí una descripción para predecir la decisión.")

st.markdown("---")
st.markdown("Desarrollado por **LTELC - Consultoría en Datos e IA** ⚙️")
