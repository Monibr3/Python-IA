```
Predicción de Citas Médicas

Proyecto en Python para predecir la demanda diaria de citas médicas por centro y especialidad, integrando preprocesamiento, generación de features, modelado y evaluación de resultados en un flujo reproducible.

__Objetivos__
- Predecir la cantidad de citas médicas diarias.
- Automatizar carga, limpieza y preprocesamiento del dataset.
- Entrenar un modelo de Machine Learning (RandomForestRegressor).
- Evaluar el modelo con métricas y visualizaciones interpretables.
- Guardar resultados y modelo para futuras predicciones.

__Estructura del Proyecto__
Prediccion_citas_medicas/
    data/
        citas_sinteticas.csv
    notebooks/
        exploracion_citas.ipynb
    src/
        __init__.py
        carga_datos.py
        limpiar_datos.py
        preprocesamiento.py
        modelo.py
        evaluacion.py
    scripts/
        generar_dataset_sintetico.py
    main.py
    requirements.txt
    README.md

__Dataset__
Columna       Descripción
fecha_cita    Fecha de la cita médica
centro_salud  Centro médico
especialidad  Especialidad de la cita
num_citas     Número de citas registradas

__Notebooks__
- exploracion_citas.ipynb: Distribución de citas, visualización de outliers y valores faltantes, primeras conclusiones.

__Código Fuente (src/)__
- carga_datos.py: Leer CSV y revisar nulos.
- limpiar_datos.py: Limpieza, reemplazo de nulos y detección de outliers.
- preprocesamiento.py: Separación de variables, train/test split, escalado.
- modelo.py: Entrenamiento y predicción con RandomForestRegressor.
- evaluacion.py: Métricas (MAE, RMSE, R²) y visualización.

__Flujo Principal (main.py)__
1. Carga de datos.
2. Limpieza y preprocesamiento.
3. División en entrenamiento y prueba.
4. Entrenamiento de pipeline con escalado y RandomForest.
5. Predicción sobre conjunto de prueba.
6. Evaluación: MAE, MSE, RMSE, R².
7. Visualización: predicciones vs reales, histograma de errores.
8. Guardado del modelo y resultados en models/.

__Instalación y Requisitos__
1. Crear entorno virtual: python -m venv venv
2. Activar entorno: Windows: venv\Scripts\activate, macOS/Linux: source venv/bin/activate
3. Instalar dependencias: pip install -r requirements.txt

Librerías principales: pandas, numpy, matplotlib, seaborn, scikit-learn, joblib, imbalanced-learn, jupyter.

__Uso__
python main.py --data data/citas_sinteticas.csv --models_dir models
- --data: Ruta al CSV.
- --models_dir: Carpeta para guardar modelo y resultados.

Archivos generados: modelo_citas.pkl, metricas.json, params.json.

__Resultados__
- Métricas: MAE, MSE, RMSE, R².
- Visualizaciones: predicciones vs reales, histograma de errores.
- Pipeline listo para futuras predicciones.

__Notas__
- Reproducible y extensible a datasets reales.
- Estructura modular: cambiar modelo, limpieza, features, métricas o visualizaciones.

Autor: Mónica Boullosa Rodríguez
Fecha: 2025
```
