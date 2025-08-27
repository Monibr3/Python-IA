import joblib
from sklearn.ensemble import RandomForestClassifier

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
    # Creamos el modelo usando los parámetros recibidos
    modelo = RandomForestClassifier(**kwargs)
    
    # Entrenamos el modelo con los datos de entrenamiento
    modelo.fit(X_train, y_train)
    
    # Devolvemos el modelo entrenado
    return modelo


def predecir(modelo, X_test):
    """
    Realiza predicciones con el modelo entrenado.
    
    Parámetros:
    - modelo (sklearn model): modelo previamente entrenado
    - X_test (array-like): variables predictoras de test
    
    Retorna:
    - y_pred (array): predicciones del modelo
    """
    
    # Se utiliza el método predict del modelo entrenado para obtener las predicciones
    return modelo.predict(X_test)



def guardar_modelo(modelo, ruta="modelo_diabetes.pkl"):
    """
    Guarda el modelo entrenado en disco usando joblib.
    
    Parámetros:
    - modelo (sklearn model): modelo a guardar
    - ruta (str): nombre del archivo de salida
    """
    
    # Guardamos el modelo en la ruta indicada usando joblib
    joblib.dump(modelo, ruta)
    
    # Mensaje indicando que el modelo se ha guardado correctamente
    print(f"Modelo guardado en {ruta}")


def cargar_modelo(ruta="modelo_diabetes.pkl"):
    """
    Carga un modelo entrenado desde disco.
    
    Parámetros:
    - ruta (str): ruta al archivo del modelo
    
    Retorna:
    - modelo cargado
    """
    
    # Cargamos el modelo desde la ruta indicada usando joblib
    modelo = joblib.load(ruta)
    
    # Mensaje indicando que el modelo se ha cargado correctamente
    print(f"Modelo cargado desde {ruta}")
    
    # Devolvemos el modelo cargado
    return modelo
