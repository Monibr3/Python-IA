import numpy as np
import matplotlib.pyplot as plt  # Para gráficos generales
import seaborn as sns            # Para visualizaciones estadísticas
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

# Configuración básica del logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


# ------------------------------------------------------------
# Función para calcular métricas básicas de regresión
# ------------------------------------------------------------
def evaluar_modelo(y_test: list, y_pred: list) -> dict:
    """
    Calcula métricas de evaluación básicas para regresión:
    MAE, MSE, RMSE y R2.
    
    Parámetros:
    - y_test (array-like): valores reales de la variable objetivo
    - y_pred (array-like): predicciones del modelo
    
    Retorna:
    - dict con métricas calculadas
    """
    # Calcular error absoluto medio (MAE)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Calcular error cuadrático medio (MSE)
    mse = mean_squared_error(y_test, y_pred)
    
    # Calcular RMSE manualmente
    rmse = np.sqrt(mse) 
    
    # Calcular coeficiente de determinación R2
    r2 = r2_score(y_test, y_pred)

    # Crear diccionario con todas las métricas
    resultados = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }

    # Mostrar por logging las métricas calculadas
    logging.info(f"Métricas calculadas: {resultados}")
    
    # Retornar diccionario con métricas
    return resultados


# ------------------------------------------------------------
# Función para graficar predicciones vs valores reales
# ------------------------------------------------------------
def graficar_predicciones(y_test: list, y_pred: list) -> None:
    """
    Grafica valores reales vs predicciones del modelo.
    Permite visualizar el desempeño del modelo de regresión.
    
    Parámetros:
    - y_test (array-like): valores reales de la variable objetivo
    - y_pred (array-like): predicciones generadas por el modelo
    """
    # Configurar tamaño de la figura
    plt.figure(figsize=(6, 4))
    
    # Crear scatterplot de valores reales vs predicciones
    sns.scatterplot(x=y_test, y=y_pred)
    
    # Dibujar línea diagonal (x=y) como referencia ideal
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    
    # Etiquetado de los ejes
    plt.xlabel("Valores Reales")
    plt.ylabel("Predicciones")
    
    # Título del gráfico
    plt.title("Predicciones vs Valores Reales")
    
    # Mostrar gráfico
    plt.show()

    # Mensaje indicando que el gráfico fue generado
    logging.info("Gráfico de predicciones vs valores reales generado")
