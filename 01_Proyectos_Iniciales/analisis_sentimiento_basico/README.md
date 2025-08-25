# Análisis de Sentimientos en Tweets

Este proyecto implementa un **modelo de análisis de sentimientos** capaz de clasificar tweets como **positivos** o **negativos**, utilizando **Python**, **procesamiento de lenguaje natural (NLP)** y **Machine Learning**.

---

## Archivos principales

- **`entrenar_modelo.py`**  
  - Carga el dataset de tweets.  
  - Aplica limpieza y preprocesamiento de texto.  
  - Transforma los tweets a vectores numéricos usando **TF-IDF**.  
  - Entrena un modelo de **Regresión Logística**.  
  - Guarda el modelo y el vectorizador para uso posterior.  
  - Incluye manejo de excepciones para asegurar robustez.

- **`main.py`**  
  - Proporciona una **interfaz de consola** para ingresar un tweet.  
  - Devuelve la predicción de sentimiento utilizando el modelo entrenado.  
  - **IMPORTANTE:** los tweets deben estar en **inglés**. Introducir tweets en otros idiomas puede producir resultados incorrectos, ya que el modelo fue entrenado únicamente con datos en inglés.

---

## Tecnologías y habilidades aplicadas

- **Python**: manipulación de datos, programación modular y buenas prácticas.  
- **Preprocesamiento de texto**: limpieza con expresiones regulares, normalización de texto.  
- **TF-IDF Vectorization**: conversión de texto a vectores numéricos.  
- **Modelos supervisados**: clasificación con **Regresión Logística** (scikit-learn).  
- **Serialización de modelos**: guardado y carga de modelos con **joblib**.  
- **Buenas prácticas**: manejo de excepciones, documentación y modularización.

---

## Uso

1. **Entrenar el modelo:**  

    ```bash
    python entrenar_modelo.py
    ```

2. **Predecir un Tweet:**  

    ```bash
    python main.py
    ```

**Nota1:** asegúrate de que los tweets introducidos estén en **inglés** para obtener predicciones fiables.


**Nota2:**
## Archivo de datos
El dataset principal utilizado se encuentra en:

`https://www.kaggle.com/datasets/kazanova/sentiment140`

> ⚠️ Este archivo no está incluido en el repositorio por su tamaño.  
> Si quieres ejecutar los scripts que dependen de él, debes descargarlo manualmente o colocarlo en la ruta indicada.
