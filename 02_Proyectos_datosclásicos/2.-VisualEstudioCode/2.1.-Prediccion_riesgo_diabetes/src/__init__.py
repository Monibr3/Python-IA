"""
Paquete principal del proyecto Predicción de Riesgo de Diabetes.

Este archivo permite importar las funciones principales directamente desde el paquete `src`.

Ejemplo de uso:
    from src import cargar_datos, entrenar_modelo
"""

# Funciones de carga de datos
from .carga_datos import cargar_datos

# Funciones de limpieza y preprocesamiento
from .limpiar_datos import limpiar_dataset, revisar_nulos, detectar_outliers
from .preprocesamiento import separar_variables, dividir_datos, escalar_datos

# Funciones de entrenamiento y predicción
from .modelo import entrenar_modelo, predecir

# Funciones de evaluación
from .evaluacion import evaluar_modelo
