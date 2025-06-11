import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from fpdf import FPDF
import base64

# Configuración inicial (debe ir primero)
st.set_page_config(page_title="VARGENTO", layout="centered")

# Cargar modelo y vectorizador
@st.cache_resource
def cargar_modelo_y_vectorizador():
    modelo = joblib.load("modelo_var_nb.pkl")
    vectorizador = joblib.load("vectorizador_var.pkl")
    return modelo, vectorizador

modelo, vectorizador = cargar_modelo_y_vectorizador()

# Interfaz
st.title("⚽ VARGENTO - Análisis Inteligente de Jugadas VAR")
st.markdown("""
### 🔹 Subí la descripción textual de la jugada
Podés escribir libremente lo que ocurrió en la jugada. Ejemplo: "El delantero remata al arco, el defensor la saca con la mano dentro del área"
""")

descripcion = st.text_area("Descripción de la jugada", height=150)

if st.button("🔢 Predecir decisión arbitral"):
    if descripcion.strip() == "":
        st.warning("Por favor ingresá una descripción.")
    else:
        descripcion_vectorizada = vectorizador.transform([descripcion])
        pred = modelo.predict(descripcion_vectorizada)[0]
        probas = modelo.predict_proba(descripcion_vectorizada)[0]
        confianza = probas[modelo.classes_.tolist().index(pred)] * 100

        st.success(f"Decisión sugerida: **{pred}** ({confianza:.2f}% confianza)")

        # Reglas relacionadas (simplificado por ahora)
        reglas_fifa = {
            "Penal": "Regla 12: Faltas e incorrecciones. Mano deliberada dentro del área por un defensor.",
            "Roja": "Regla 12: Juego brusco grave, conducta violenta, impedir un gol con mano intencional.",
            "Amarilla": "Regla 12: Conducta antideportiva, protestas reiteradas, demorar el juego.",
            "Offside": "Regla 11: Posición adelantada al momento de recibir el balón.",
            "Gol anulado": "Regla 11 o 12: Offside o falta previa antes del gol.",
            "Gol válido": "Regla 10: El gol es válido si no hay infracciones.",
            "No penal": "Regla 12: No hay contacto o falta suficiente dentro del área."
        }

        if pred in reglas_fifa:
            st.info(f"Referencia reglamentaria: {reglas_fifa[pred]}")

        # PDF opcional
        if st.button("📄 Generar informe PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, f"Jugada: {descripcion}\n\nDecisión sugerida: {pred} ({confianza:.2f}% confianza)\n\nReglamento: {reglas_fifa.get(pred, 'No disponible')}")
            pdf_output = "reporte_var.pdf"
            pdf.output(pdf_output)

            with open(pdf_output, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
                href = f'<a href="data:application/octet-stream;base64,{b64}" download="reporte_var.pdf">Descargar PDF</a>'
                st.markdown(href, unsafe_allow_html=True)

# Footer
st.markdown("""
---
<div style='text-align: center; color: gray;'>
Desarrollado por **LTELC - Consultoría en Datos e IA** ⚙️
</div>
""", unsafe_allow_html=True)

