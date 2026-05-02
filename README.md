# 🍷 Clasificación de Vinos con Machine Learning

## 📌 Descripción

Este proyecto implementa un pipeline básico de Machine Learning para la clasificación de vinos utilizando el dataset **Wine Dataset** de scikit-learn.

El objetivo principal es transformar un análisis monolítico en un proyecto modular, separando las etapas de análisis, entrenamiento y evaluación en diferentes archivos.

---

## 🧠 Objetivos

* Realizar análisis exploratorio de datos (EDA)
* Entrenar modelos de clasificación
* Evaluar el desempeño de los modelos
* Implementar una estructura modular del proyecto

---

## 📂 Estructura del proyecto

```
clasificacion-vinos/
│
├── src/
│   ├── eda.py              # Análisis exploratorio de datos
│   ├── entrenamiento.py    # Entrenamiento de modelos
│   ├── prueba.py           # Evaluación del modelo
│
├── modelo.pkl              # Modelo entrenado
├── scaler.pkl              # Escalador de datos
└── README.md
```

---

## 📊 Dataset

Se utiliza el dataset **Wine Dataset** incluido en la librería scikit-learn.

Contiene características químicas de distintos vinos, como:

* alcohol
* pH
* flavonoides
* intensidad de color

Y una variable objetivo (`target`) que indica la clase del vino.

---

## ⚙️ Tecnologías utilizadas

* Python
* pandas
* scikit-learn
* matplotlib
* seaborn
* joblib

---

## 🚀 Cómo ejecutar el proyecto

1. Clonar el repositorio:

```
git clone https://github.com/tu-usuario/clasificacion-vinos.git
cd clasificacion-vinos
```

2. Instalar dependencias:

```
pip install pandas scikit-learn matplotlib seaborn joblib
```

3. Ejecutar los scripts en orden:

```
python src/eda.py
python src/entrenamiento.py
python src/prueba.py
```

---

## 📈 Modelos utilizados

* Random Forest
* Support Vector Machine (SVM)
* Logistic Regression

Se selecciona el mejor modelo en función de su desempeño (accuracy).

---

## 📊 Métricas evaluadas

* Accuracy
* Precision
* Recall
* F1-score
* Matriz de confusión

---

## ✨ Resultados

El modelo logra clasificar correctamente los vinos con un buen nivel de precisión, demostrando la efectividad de los algoritmos utilizados.

---

## ⚠️ Problemas encontrados

Durante el desarrollo se presentaron algunos retos como:

* Comprender la estructura del proyecto original
* Manejar la persistencia del modelo con joblib
* Organizar correctamente el flujo de ejecución

---

## 💡 Conclusión

Se logró construir un proyecto modular y funcional de Machine Learning, aplicando buenas prácticas de organización de código y separación de responsabilidades.

---

## 👩‍💻 Autor

Iris Pérez
