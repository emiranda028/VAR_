# ⚽ VARGENTO – Asistente VAR Inteligente

Bienvenido a **VARGENTO**, una aplicación desarrollada con Streamlit que analiza descripciones de jugadas de fútbol y sugiere decisiones arbitrales basadas en modelos entrenados con más de 2000 jugadas reales.

## 🎯 Funcionalidades

- Escribir una descripción de jugada
- Subir imagen o video
- Pegar un link de YouTube
- Recibir sugerencia de decisión arbitral
- Ver la regla FIFA aplicada
- Ver porcentaje de confianza del modelo

## 📦 Archivos

- `app.py`: App principal en Streamlit
- `modelo_var_nb.pkl`: Modelo entrenado
- `vectorizador_var.pkl`: Vectorizador TF-IDF
- `label_encoder_var.pkl`: Codificador de etiquetas
- `EPL_VAR_2000_jugadas.csv`: Dataset original

## 🛠 Requisitos

```bash
pip install -r requirements.txt
```

## 🚀 Cómo correrlo

```bash
streamlit run app.py
```

## 🧠 Entrenamiento

El modelo fue entrenado con datos reales y etiquetas verificadas. Si querés contribuir con más jugadas etiquetadas, abrí un issue o hacé un pull request.

## 👨‍💻 Desarrollado por

**LTELC – Consultoría en Datos e IA**
