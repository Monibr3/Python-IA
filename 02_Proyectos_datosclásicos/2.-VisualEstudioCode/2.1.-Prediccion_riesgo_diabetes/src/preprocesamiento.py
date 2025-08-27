import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def separar_variables(df, columna_objetivo="Outcome"):
    """
    Separa las variables de entrenamiento (X) de la variable objetivo (y).
    
    Parámetros:
    - df (pd.DataFrame): dataset limpio
    - columna_objetivo (str): nombre de la columna objetivo
    
    Retorna:
    - X (pd.DataFrame)
    - y (pd.Series)
    """
    
    # Comprobamos que la columna objetivo exista en el DataFrame
    if columna_objetivo not in df.columns:
        # Si no existe, lanzamos un error indicando que la columna no está presente
        raise ValueError(f"La columna objetivo '{columna_objetivo}' no está en el dataset")
    
    # X contiene todas las columnas excepto la columna objetivo
    X = df.drop(columns=[columna_objetivo])
    
    # y contiene únicamente la columna objetivo
    y = df[columna_objetivo]
    
    # Devolvemos X (features) e y (target)
    return X, y

def dividir_datos(X, y, test_size=0.2, random_state=42):
    """
    Divide el dataset en conjuntos de entrenamiento y prueba (train y test).
    
    Parámetros:
    - X (pd.DataFrame): variables independientes (features)
    - y (pd.Series): variable objetivo (target)
    - test_size (float): proporción del conjunto de test respecto al total (por defecto 0.2)
    - random_state (int): semilla para asegurar reproducibilidad
    
    Retorna:
    - X_train, X_test, y_train, y_test
    """
    
    # Utilizamos train_test_split de scikit-learn para dividir los datos
    # stratify=y asegura que la proporción de clases en y se mantenga en ambos conjuntos
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def escalar_datos(X_train, X_test):
    """
    Escala los datos con StandardScaler (media 0, varianza 1).
    
    Parámetros:
    - X_train (pd.DataFrame): conjunto de entrenamiento
    - X_test (pd.DataFrame): conjunto de prueba
    
    Retorna:
    - X_train_scaled (np.array): datos de entrenamiento escalados
    - X_test_scaled (np.array): datos de prueba escalados
    - scaler (StandardScaler): objeto scaler, útil para aplicar la misma transformación a nuevos datos
    """
    
    # Creamos el objeto StandardScaler de scikit-learn
    scaler = StandardScaler()
    
    # Ajustamos el scaler a los datos de entrenamiento y transformamos X_train
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Transformamos X_test usando los parámetros calculados en X_train
    X_test_scaled = scaler.transform(X_test)
    
    # Devolvemos los conjuntos escalados y el objeto scaler
    return X_train_scaled, X_test_scaled, scaler
