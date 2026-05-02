# src/prueba.py

import pandas as pd
import joblib
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

def cargar_datos():
    data = load_wine()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    return X, y

if __name__ == "__main__":
    X, y = cargar_datos()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = joblib.load("scaler.pkl")
    modelo = joblib.load("modelo.pkl")

    X_test = scaler.transform(X_test)

    y_pred = modelo.predict(X_test)

    print("\n MATRIZ DE CONFUSIÓN:")
    print(confusion_matrix(y_test, y_pred))

    print("\n REPORTE:")
    print(classification_report(y_test, y_pred))