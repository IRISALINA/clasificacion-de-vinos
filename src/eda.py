# src/eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine

def cargar_datos():
    data = load_wine()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df

def analizar_datos(df):
    print("\n Información general:")
    print(df.info())

    print("\n Estadísticas descriptivas:")
    print(df.describe())

def visualizar_datos(df):
    # Histogramas
    df.hist(figsize=(12,10))
    plt.suptitle("Histogramas")
    plt.show()

    # Boxplot
    plt.figure(figsize=(12,6))
    sns.boxplot(data=df)
    plt.title("Boxplot")
    plt.xticks(rotation=90)
    plt.show()

    # Correlación
    plt.figure(figsize=(10,8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title("Matriz de correlación")
    plt.show()

if __name__ == "__main__":
    df = cargar_datos()
    analizar_datos(df)
    visualizar_datos(df)