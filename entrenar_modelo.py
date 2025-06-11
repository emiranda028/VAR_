import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

df = pd.read_csv("VAR_Limpio_Generado.csv")
vectorizador = CountVectorizer()
X = vectorizador.fit_transform(df["descripcion"])
le = LabelEncoder()
y = le.fit_transform(df["Decision"])

modelo = XGBClassifier(eval_metric="mlogloss", n_estimators=10, max_depth=3)
modelo.fit(X, y)

with open("modelo.pkl", "wb") as f:
    pickle.dump(modelo, f)

with open("vectorizador.pkl", "wb") as f:
    pickle.dump(vectorizador, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("✅ Modelo y recursos guardados")
