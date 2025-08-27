# Predicción de Riesgo de Diabetes

Este proyecto implementa un modelo de Machine Learning para predecir el riesgo de **diabetes** a partir de variables clínicas del dataset **Pima Indians Diabetes Database**.

El flujo incluye:

1. Exploración de datos (EDA)
2. Preprocesamiento y limpieza
3. Balanceo de clases con SMOTE
4. Entrenamiento con Random Forest
5. Evaluación con métricas y curva ROC
6. Guardado del modelo para uso posterior

---

## Estructura del Proyecto

```
Prediccion_riesgo_diabetes/
│
├── data/                     # Carpeta para tus datasets CSV
│   └── diabetes.csv          # Dataset de diabetes
│
├── notebooks/                # Para experimentación o EDA
│   └── exploracion.ipynb     # Notebook de análisis exploratorio
│
├── src/                      # Código Python modular del proyecto
│   ├── __init__.py           # Permite importar funciones de src directamente
│   ├── carga_datos.py        # Funciones para cargar el dataset
│   ├── limpiar_datos.py      # Funciones para limpiar y revisar datos
│   ├── preprocesamiento.py   # Separar variables, dividir y escalar
│   ├── modelo.py             # Entrenamiento y predicción del modelo
│   └── evaluacion.py         # Funciones para evaluar el modelo
│
├── main.py                   # Archivo principal para ejecutar el flujo completo
├── requirements.txt          # Librerías necesarias para el proyecto
└── README.md                 # Documentación profesional del proyecto
```

---

## Instalación y Uso

1. Clonar el repositorio:

```bash
git clone 
cd Prediccion_riesgo_diabetes
```

2. Crear y activar entorno virtual (opcional pero recomendado):

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

3. Instalar dependencias:

```bash
pip install -r requirements.txt
```

4. Ejecutar el flujo principal:

```bash
python main.py
```

---

## Resultados del Modelo

* Accuracy: \~0.75
* Precision (Diabetes): \~0.61
* Recall (Diabetes): \~0.76
* AUC: \~0.84

El modelo logra un buen balance entre sensibilidad y precisión, priorizando la detección de posibles casos de diabetes.

---

## Visualizaciones

Durante el análisis y la evaluación se generan:

* Histogramas de variables clínicas
* Matriz de correlación
* Boxplots de outliers
* Matriz de confusión
* Curva ROC

---

## Futuras Mejoras

* Probar modelos alternativos (XGBoost, LightGBM)
* Optimización de hiperparámetros (GridSearchCV / Optuna)
* Implementación de API para predicciones en tiempo real
* Dashboard interactivo (Streamlit / Dash)

---

## Autora

Proyecto desarrollado por Mónica Boullosa Rodríguez

