import streamlit as st

# ✅ La primera línea debe ser la configuración de página
st.set_page_config(page_title="Predicción VAR", layout="centered")

import joblib
import numpy as np
from PIL import Image

# Cargar modelo y recursos
modelo = joblib.load("modelo_var_nb.pkl")
vectorizador = joblib.load("vectorizador_var.pkl")
le = joblib.load("label_encoder_var.pkl")

# UI principal
st.image("https://media.tenor.com/xOb4uwv-VV8AAAAC/var-checking.gif", use_container_width=True)
st.title("⚽ Bienvenido a VARGENTO - Asistente VAR Inteligente")
st.markdown("""
Subí una descripción textual de una jugada para que el sistema sugiera una decisión según el reglamento FIFA.

👉 También podés subir una imagen o video de la jugada.  
👉 O pegar el link de YouTube si lo tenés.  
👉 Recibirás una sugerencia de decisión acompañada de la regla FIFA correspondiente.

📖 [Ver Reglamento de Juego FIFA](https://digitalhub.fifa.com/m/799749e5f64c0f86/original/lnc9zjo8xf2j3nvwfazh-pdf.pdf)
""")

# Inputs del usuario
descripcion = st.text_area("📝 Describí la jugada con claridad")
archivo_subido = st.file_uploader("📎 Subí imagen o video (opcional):", type=["jpg", "jpeg", "png", "mp4"])
link_youtube = st.text_input("📺 Pegá link de YouTube (opcional):")

# Predicción
if st.button("🔍 Predecir decisión"):
    if descripcion.strip() == "":
        st.warning("Por favor, ingresá una descripción válida.")
    else:
        try:
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

            # Mostrar multimedia
            if archivo_subido:
                if archivo_subido.type.startswith("video"):
                    st.video(archivo_subido)
                elif archivo_subido.type.startswith("image"):
                    img = Image.open(archivo_subido)
                    st.image(img, caption="Imagen de la jugada", use_container_width=True)

            if link_youtube:
                st.video(link_youtube)

        except Exception as e:
            st.error(f"❌ Error en la predicción: {e}")

# Pie de página
st.markdown("---")
st.markdown('<div style="text-align: center; color: gray;">Desarrollado por LTELC - Consultoría en Datos e IA ⚙️</div>', unsafe_allow_html=True)
