"""
Paquete principal del proyecto Predicción de Citas Médicas.

Este archivo permite importar las funciones principales directamente desde el paquete `src`.

Ejemplo de uso:
    from src import cargar_datos, entrenar_modelo
    from src import limpiar_dataset, separar_variables
"""

# Funciones de carga de datos
from .carga_datos import cargar_datos

# Funciones de limpieza y preprocesamiento
# Ahora adaptadas para el dataset de citas médicas sintéticas
from .limpiar_datos import limpiar_dataset, revisar_nulos, detectar_outliers
from .preprocesamiento import separar_variables, dividir_datos, escalar_datos

# Funciones de entrenamiento y predicción
# Pueden usarse para modelos de predicción de demanda de citas
from .modelo import entrenar_modelo, predecir

# Funciones de evaluación
# Evaluación de modelos con métricas y visualizaciones
from .evaluacion import evaluar_modelo
