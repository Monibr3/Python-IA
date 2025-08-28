"""
main.py - Flujo principal para Predicción de Citas Médicas

Este script integra todas las etapas:
1. Carga y limpieza de datos
2. Preprocesamiento
3. Entrenamiento del modelo
4. Predicción
5. Evaluación del rendimiento del modelo
6. Guardado del modelo entrenado, métricas y parámetros
7. Visualización de resultados
"""
import os
import json
import logging
import argparse
import matplotlib.pyplot as plt
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline  # Mantener pipeline
from src.carga_datos import cargar_datos
from src.limpiar_datos import limpiar_dataset, revisar_nulos, detectar_outliers
from src.preprocesamiento import separar_variables, dividir_datos
from src.evaluacion import evaluar_modelo

# Configurar logging para salida informativa en consola
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def graficar_predicciones(y_test, y_pred):
    """
    Gráfico de predicciones vs valores reales
    """
    plt.figure(figsize=(6, 4))
    plt.scatter(y_test, y_pred, alpha=0.5, color="royalblue")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    plt.xlabel("Valores reales (y_test)")
    plt.ylabel("Predicciones (y_pred)")
    plt.title("Predicciones vs Valores Reales")
    plt.show()


def main():
    # --------------------------------------------------------
    # Configuración de argumentos para flexibilidad de rutas
    # --------------------------------------------------------
    parser = argparse.ArgumentParser(description="Predicción de Citas Médicas")
    parser.add_argument(
        "--data", type=str, default="data/citas_sinteticas.csv", help="Ruta al archivo CSV de datos"
    )
    parser.add_argument(
        "--models_dir", type=str, default="models", help="Carpeta para guardar modelos y resultados"
    )
    args = parser.parse_args()

    # --------------------------------------------------------
    # 1. Cargar datos desde CSV
    # --------------------------------------------------------
    logging.info("Cargando datos desde: %s", args.data)
    df = cargar_datos(args.data)
    if df is None:
        raise FileNotFoundError("No se encontró el archivo de datos. Verifica la ruta y que el CSV exista.")

    # --------------------------------------------------------
    # 2. Revisar y limpiar datos
    # --------------------------------------------------------
    logging.info("Revisando valores nulos y consistencia del dataset")
    revisar_nulos(df)

    logging.info("Detectando outliers en columnas numéricas del dataset")
    detectar_outliers(df, columnas=["num_citas"], graficar=True)

    logging.info("Limpiando dataset: imputación de valores faltantes según estrategia definida")
    df = limpiar_dataset(df, columnas_nulos=[])

    # --------------------------------------------------------
    # 3. Preprocesamiento
    # --------------------------------------------------------
    logging.info("Separando variables predictoras (X) y objetivo (y)")
    X, y = separar_variables(df, columna_objetivo="num_citas")

    logging.info("Dividiendo dataset en entrenamiento y prueba (train/test split)")
    X_train, X_test, y_train, y_test = dividir_datos(X, y)

    # --------------------------------------------------------
    # 4. Pipeline con escalado + RandomForestRegressor
    # --------------------------------------------------------
    logging.info("Creando pipeline con StandardScaler y RandomForestRegressor")
    model_params = {
        "n_estimators": 500,
        "max_depth": 6,
        "min_samples_split": 4,
        "min_samples_leaf": 3,
        "max_features": "sqrt",
        "random_state": 42
    }

    pipeline = ImbPipeline(steps=[
        ("scaler", StandardScaler()),                       # Escalado de features
        ("regressor", RandomForestRegressor(**model_params))  # Modelo de regresión
    ])

    logging.info("Entrenando pipeline sobre datos de entrenamiento")
    pipeline.fit(X_train, y_train)

    # --------------------------------------------------------
    # 5. Predicciones sobre conjunto de prueba
    # --------------------------------------------------------
    logging.info("Generando predicciones sobre conjunto de prueba")
    y_pred = pipeline.predict(X_test)

    # --------------------------------------------------------
    # 6. Evaluación del modelo
    # --------------------------------------------------------
    logging.info("Evaluando métricas de rendimiento: RMSE, MAE, R2")
    metricas = evaluar_modelo(y_test, y_pred)
    logging.info("Resultados: %s", metricas)

    # --------------------------------------------------------
    # 7. Guardar modelo y resultados
    # --------------------------------------------------------
    logging.info("Comprobando existencia de carpeta para guardar modelos y resultados")
    if not os.path.exists(args.models_dir):
        os.makedirs(args.models_dir)

    # Guardar pipeline completo
    dump(pipeline, os.path.join(args.models_dir, "modelo_citas.pkl"))

    # Guardar métricas en JSON
    with open(os.path.join(args.models_dir, "metricas.json"), "w") as f:
        json.dump(metricas, f, indent=4)

    # Guardar hiperparámetros en JSON
    with open(os.path.join(args.models_dir, "params.json"), "w") as f:
        json.dump(model_params, f, indent=4)

    logging.info("Pipeline, métricas y parámetros guardados correctamente")

    # --------------------------------------------------------
    # 8. Visualización de resultados
    # --------------------------------------------------------
    logging.info("Generando visualizaciones de resultados")
    
    # Gráfico Predicciones vs Valores Reales
    graficar_predicciones(y_test, y_pred)

    # Histograma de errores (residuos)
    errores = y_test - y_pred
    plt.figure(figsize=(6, 4))
    plt.hist(errores, bins=30, color='skyblue', edgecolor='black')
    plt.title("Distribución de errores (residuos)")
    plt.xlabel("Error (y_test - y_pred)")
    plt.ylabel("Frecuencia")
    plt.show()

    logging.info("Visualizaciones completadas")
    logging.info("Flujo completado con éxito.")


# --------------------------------------------------------
# Ejecutar main solo si se invoca directamente
# --------------------------------------------------------
if __name__ == "__main__":
    main()
