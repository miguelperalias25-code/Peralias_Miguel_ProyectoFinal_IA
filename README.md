# UFC Fight Predictor (IA)

Proyecto de automatización inteligente basado en Machine Learning que predice el ganador de un combate de UFC a partir de estadísticas históricas de los peleadores.

## Descripción

Este sistema utiliza modelos de Machine Learning para analizar datos históricos de peleas y características de luchadores, con el objetivo de estimar la probabilidad de victoria entre dos peleadores.

El sistema incluye:

* Procesamiento de datos
* Entrenamiento de modelos
* Interfaz visual con Streamlit

## Tecnologías utilizadas

* Python
* Pandas / NumPy
* Scikit-learn
* Streamlit
* Matplotlib / Seaborn

## Dataset

Se utilizan dos datasets principales:

* `ufc_gold_dataset_final.csv` → historial de peleas
* `ufc_fighters_final.csv` → estadísticas de peleadores

## Funcionamiento
El sistema sigue estos pasos:

1. Limpieza y transformación de datos
2. Creación de variables (features)
3. Entrenamiento de modelos ML
4. Predicción de probabilidades
5. Visualización en una app web

## Ejecución del proyecto

## 1. Preparar datos
```
python preparar_datos.py
```
## 2. Entrenar modelo
```
Ejecutar en google collab por ejemplo Ufc_prediccion.ipynb
```
## 3. Ejecutar app
```
streamlit run app.py
```
## Resultados

* Modelo mejorado con AUC ≈ **0.79**
* Mejora respecto a versión anterior (0.71)

## Objetivo
Automatizar el análisis de combates de UFC usando IA, reduciendo el análisis manual y proporcionando predicciones basadas en datos.

## Autor
Miguel Peralais Lopez

