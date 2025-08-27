"""
main.py - Flujo principal para Predicción de Riesgo de Diabetes

Este script integra todas las etapas:
1. Carga y limpieza de datos
2. Preprocesamiento
3. Entrenamiento del modelo (con balance de clases mediante SMOTE)
4. Predicción
5. Evaluación del rendimiento del modelo
6. Guardado del modelo entrenado, métricas y parámetros
"""
import os
import json
import logging
import numpy as np
import argparse
from joblib import dump
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from src.carga_datos import cargar_datos
from src.limpiar_datos import limpiar_dataset, revisar_nulos, detectar_outliers
from src.preprocesamiento import separar_variables, dividir_datos
from src.evaluacion import evaluar_modelo, mostrar_matriz_confusion, informe_clasificacion, curva_roc

# Configurar logging para salida informativa en consola
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    # Argumentos para flexibilidad de rutas
    parser = argparse.ArgumentParser(description="Predicción de Riesgo de Diabetes")
    parser.add_argument("--data", type=str, default="data/diabetes.csv", help="Ruta al archivo CSV de datos")
    parser.add_argument("--models_dir", type=str, default="models", help="Carpeta para guardar modelos y resultados")
    args = parser.parse_args()

    # 1. Cargar datos desde CSV
    logging.info("Cargando datos desde: %s", args.data)
    df = cargar_datos(args.data)
    if df is None:
        raise FileNotFoundError("No se encontró el archivo de datos. Verifica la ruta y que el CSV exista.")

    # 2. Revisar y limpiar datos
    logging.info("Revisando valores nulos y consistencia del dataset")
    revisar_nulos(df)

    logging.info("Detectando outliers en columnas críticas (Glucose, BloodPressure, BMI, Insulin)")
    detectar_outliers(df, columnas=["Glucose", "BloodPressure", "BMI", "Insulin"], graficar=True)

    logging.info("Eliminando o imputando valores faltantes según estrategia definida en limpiar_dataset")
    df = limpiar_dataset(df, columnas_nulos=["Glucose", "BloodPressure", "BMI", "Insulin"])

    # 3. Preprocesamiento
    logging.info("Separando variables predictoras (X) y objetivo (y)")
    X, y = separar_variables(df, columna_objetivo="Outcome")

    logging.info("Dividiendo dataset en entrenamiento y prueba (train/test split)")
    X_train, X_test, y_train, y_test = dividir_datos(X, y)

    # 4. Pipeline con SMOTE + escalado + modelo
    logging.info(
        "Creando pipeline que incluye balanceo de clases con SMOTE, escalado de features, "
        "y RandomForestClassifier con hiperparámetros definidos"
    )
    classifier_params = {
        "n_estimators": 500,
        "max_depth": 6,
        "min_samples_split": 4,
        "min_samples_leaf": 3,
        "max_features": "sqrt",
        "random_state": 42,
        "class_weight": "balanced"
    }

    pipeline = ImbPipeline(steps=[
        ("smote", SMOTE(random_state=42)),          # Balancear clases minoritarias
        ("scaler", StandardScaler()),               # Escalado de features
        ("classifier", RandomForestClassifier(**classifier_params))
    ])

    # Entrenar modelo completo con pipeline
    logging.info("Entrenando pipeline sobre datos de entrenamiento")
    pipeline.fit(X_train, y_train)

    # 5. Predicciones
    logging.info("Generando predicciones sobre conjunto de prueba")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]  # Probabilidades para ROC

    # 6. Evaluación del modelo
    logging.info("Evaluando métricas de rendimiento: precisión, recall, F1-score")
    metricas = evaluar_modelo(y_test, y_pred)
    logging.info("Resultados: %s", metricas)

    logging.info("Mostrando informe de clasificación detallado")
    informe_clasificacion(y_test, y_pred)

    logging.info("Mostrando matriz de confusión")
    mostrar_matriz_confusion(y_test, y_pred)

    logging.info("Calculando curva ROC y AUC")
    auc_score = curva_roc(y_test, y_proba)
    logging.info("AUC: %.3f", auc_score)

    # Agregar AUC a métricas
    metricas["auc"] = auc_score

    # 7. Guardar modelo y resultados
    logging.info("Comprobando existencia de carpeta para guardar modelos y resultados")
    if not os.path.exists(args.models_dir):
        os.makedirs(args.models_dir)

    # Guardar pipeline completo
    dump(pipeline, os.path.join(args.models_dir, "modelo_diabetes.pkl"))

    # Guardar métricas en JSON
    with open(os.path.join(args.models_dir, "metricas.json"), "w") as f:
        json.dump(metricas, f, indent=4)

    # Guardar hiperparámetros en JSON
    with open(os.path.join(args.models_dir, "params.json"), "w") as f:
        json.dump(classifier_params, f, indent=4)

    logging.info("Pipeline, métricas y parámetros guardados correctamente")
    logging.info("Flujo completado con éxito.")


if __name__ == "__main__":
    main()
