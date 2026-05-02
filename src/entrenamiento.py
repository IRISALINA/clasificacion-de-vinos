# src/entrenamiento.py

import pandas as pd
import joblib
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def cargar_datos():
    data = load_wine()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    return X, y

def entrenar_modelos(X_train, y_train):
    modelos = {
        "RandomForest": RandomForestClassifier(),
        "SVM": SVC(probability=True),
        "LogisticRegression": LogisticRegression(max_iter=1000)
    }

    resultados = []
    mejor_modelo = None
    mejor_score = 0

    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        print(f"\nModelo: {nombre}")
        print(f"Accuracy: {acc:.4f}")

        resultados.append((nombre, acc))

        if acc > mejor_score:
            mejor_score = acc
            mejor_modelo = modelo

    return mejor_modelo

if __name__ == "__main__":
    X, y = cargar_datos()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    modelo = entrenar_modelos(X_train, y_train)

    joblib.dump(modelo, "modelo.pkl")
    joblib.dump(scaler, "scaler.pkl")

    print("\n Modelo guardado")