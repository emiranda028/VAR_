import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import plotly.express as px
from PIL import Image
import io
import base64
from fpdf import FPDF

st.set_page_config(page_title="VARGENTO - Análisis VAR", layout="wide")

st.markdown("""
<style>
    .big-font { font-size:30px !important; font-weight: bold; color: #003366; }
    .medium-font { font-size:20px !important; color: #444; }
    .footer { text-align: center; color: gray; margin-top: 50px; font-size: 13px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-font">⚽ VARGENTO - Análisis Inteligente de Jugadas VAR</div>', unsafe_allow_html=True)
st.markdown('<div class="medium-font">Subí una jugada, escribí la descripción y recibí la sugerencia del sistema.</div>', unsafe_allow_html=True)
st.markdown("---")

archivo = st.file_uploader("📂 Subí el archivo CSV (debe tener columnas 'descripcion' y 'Decision')", type="csv")

@st.cache_data
def entrenar_modelo(df):
    df = df.dropna(subset=["descripcion", "Decision"])
    df = df[df["descripcion"].str.len() > 5]
    conteo = df["Decision"].value_counts()
    clases_validas = conteo[conteo >= 10].index.tolist()
    df = df[df["Decision"].isin(clases_validas)]

    if len(df["Decision"].unique()) < 2:
        st.error("❌ Se requieren al menos 2 clases válidas.")
        st.stop()

    vectorizador = CountVectorizer()
    X = vectorizador.fit_transform(df["descripcion"])

    le = LabelEncoder()
    y = le.fit_transform(df["Decision"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    modelo = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return modelo, vectorizador, le, acc, df

if archivo is not None:
    df = pd.read_csv(archivo)
    if "descripcion" not in df.columns or "Decision" not in df.columns:
        st.error("❌ El CSV debe tener columnas llamadas 'descripcion' y 'Decision'.")
    else:
        modelo, vectorizador, le, acc, df_train = entrenar_modelo(df)
        st.success(f"✅ Modelo entrenado con {len(df_train)} jugadas - Precisión: {acc*100:.2f}%")

        st.markdown("## 🎬 Ingresá una nueva jugada")
        descripcion = st.text_area("📝 Describí la jugada", "Jugador comete falta dentro del área...")
        archivo_media = st.file_uploader("📸 Subí una imagen o video (opcional)", type=["jpg", "png", "jpeg", "mp4"])
        youtube_link = st.text_input("🔗 Link de YouTube (opcional)")

        if st.button("🔍 Predecir decisión"):
            if descripcion.strip() == "":
                st.warning("Ingresá una descripción válida.")
            else:
                X_nueva = vectorizador.transform([descripcion])
                proba = modelo.predict_proba(X_nueva)[0]
                pred = modelo.predict(X_nueva)[0]
                decision = le.inverse_transform([pred])[0]
                confianza = proba[pred] * 100
                st.success(f"📢 Decisión sugerida: **{decision}** ({confianza:.2f}% de confianza)")

                st.markdown("#### 📊 Detalle de probabilidades")
                df_prob = pd.DataFrame({
                    "Decisión": le.inverse_transform(range(len(proba))),
                    "Probabilidad (%)": proba * 100
                })
                fig = px.bar(df_prob, x="Decisión", y="Probabilidad (%)", title="Distribución de probabilidades")
                st.plotly_chart(fig)

                if archivo_media:
                    if archivo_media.type.startswith("image"):
                        st.image(Image.open(archivo_media), caption="Imagen de la jugada", use_column_width=True)
                    elif archivo_media.type.startswith("video"):
                        st.video(archivo_media)
                elif youtube_link:
                    st.video(youtube_link)

                # PDF
                if st.button("📄 Descargar reporte PDF"):
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    pdf.multi_cell(0, 10, f"Jugada: {descripcion}

Decisión sugerida: {decision}
Confianza: {confianza:.2f}%")
                    pdf_output = io.BytesIO()
                    pdf.output(pdf_output)
                    pdf_output.seek(0)
                    b64 = base64.b64encode(pdf_output.read()).decode('utf-8')
                    href = f'<a href="data:application/octet-stream;base64,{b64}" download="reporte_var.pdf">📥 Descargar PDF</a>'
                    st.markdown(href, unsafe_allow_html=True)

        st.markdown("### 📈 Distribución de decisiones en el dataset")
        st.bar_chart(df_train["Decision"].value_counts())

        st.markdown('<div class="footer">Desarrollado por LTELC - Consultoría en Datos e IA ⚙️</div>', unsafe_allow_html=True)
