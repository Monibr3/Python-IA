import joblib
from sklearn.ensemble import RandomForestClassifier
import logging

# Configuración básica del logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ------------------------------------------------------------
# Función para entrenar un modelo RandomForest
# ------------------------------------------------------------
def entrenar_modelo(X_train, y_train, **kwargs):
    """
    Entrena un modelo RandomForestClassifier con parámetros opcionales.
    
    Parámetros:
    - X_train: variables predictoras de entrenamiento
    - y_train: etiquetas de entrenamiento
    - **kwargs: cualquier parámetro de RandomForestClassifier
    
    Retorna:
    - modelo entrenado
    """
    # Crear el modelo con los parámetros opcionales
    modelo = RandomForestClassifier(**kwargs)
    
    # Entrenar el modelo con los datos de entrenamiento
    modelo.fit(X_train, y_train)
    
    logging.info("Modelo RandomForest entrenado correctamente")
    
    return modelo

# ------------------------------------------------------------
# Función para predecir clases
# ------------------------------------------------------------
def predecir(modelo, X_test):
    """
    Realiza predicciones de clase con el modelo entrenado.
    
    Parámetros:
    - modelo (sklearn model): modelo previamente entrenado
    - X_test (array-like): variables predictoras de test
    
    Retorna:
    - y_pred (array): predicciones del modelo
    """
    # Devolver las predicciones de clase
    y_pred = modelo.predict(X_test)
    logging.info(f"Predicciones de clase realizadas sobre {X_test.shape[0]} muestras")
    return y_pred

# ------------------------------------------------------------
# Función para predecir probabilidades
# ------------------------------------------------------------
def predecir_proba(modelo, X_test):
    """
    Realiza predicciones de probabilidad con el modelo entrenado.
    Útil para curvas ROC y cálculo de AUC.
    
    Parámetros:
    - modelo (sklearn model): modelo previamente entrenado
    - X_test (array-like): variables predictoras de test
    
    Retorna:
    - y_proba (array): probabilidades estimadas de la clase positiva
    """
    # Obtener la probabilidad de la clase positiva (1)
    y_proba = modelo.predict_proba(X_test)[:, 1]
    logging.info(f"Predicciones de probabilidad realizadas sobre {X_test.shape[0]} muestras")
    return y_proba

# ------------------------------------------------------------
# Función para guardar el modelo entrenado
# ------------------------------------------------------------
def guardar_modelo(modelo, ruta="modelo_diabetes.pkl"):
    """
    Guarda el modelo entrenado en disco usando joblib.
    
    Parámetros:
    - modelo (sklearn model): modelo a guardar
    - ruta (str): nombre del archivo de salida
    """
    # Guardar el modelo en el archivo indicado
    joblib.dump(modelo, ruta)
    logging.info(f"Modelo guardado en {ruta}")

# ------------------------------------------------------------
# Función para cargar un modelo entrenado desde disco
# ------------------------------------------------------------
def cargar_modelo(ruta="modelo_diabetes.pkl"):
    """
    Carga un modelo entrenado desde disco.
    Maneja errores si el archivo no existe.
    
    Parámetros:
    - ruta (str): ruta al archivo del modelo
    
    Retorna:
    - modelo cargado (sklearn model) o None si falla
    """
    try:
        # Intentar cargar el modelo
        modelo = joblib.load(ruta)
        logging.info(f"Modelo cargado desde {ruta}")
        return modelo
    except FileNotFoundError:
        # Mensaje de error si el archivo no existe
        logging.error(f"No se encontró el archivo {ruta}. Comprueba la ruta.")
        return None
