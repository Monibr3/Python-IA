import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

# Configuración básica del logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ------------------------------------------------------------
# Función para separar variables predictoras de la variable objetivo
# ------------------------------------------------------------
def separar_variables(df, columna_objetivo="Outcome"):
    """
    Separa las variables de entrenamiento (X) de la variable objetivo (y),
    convirtiendo fechas y categóricas a numéricas automáticamente.
    
    Parámetros:
    - df (pd.DataFrame): dataset limpio
    - columna_objetivo (str): nombre de la columna objetivo
    
    Retorna:
    - X (pd.DataFrame)
    - y (pd.Series)
    
    Lanza:
    - ValueError si la columna objetivo no existe en el DataFrame
    """
    # Verificar que la columna objetivo exista en el DataFrame
    if columna_objetivo not in df.columns:
        raise ValueError(f"La columna objetivo '{columna_objetivo}' no está en el dataset")
    
    # Seleccionar todas las columnas excepto la objetivo como variables predictoras
    X = df.drop(columns=[columna_objetivo])
    
    # Seleccionar la columna objetivo
    y = df[columna_objetivo]
    
    # Convertir columnas de fecha a número de días desde 1970-01-01
    for col in X.columns:
        if pd.api.types.is_datetime64_any_dtype(X[col]):
            X[col] = (X[col] - pd.Timestamp("1970-01-01")).dt.days
        elif X[col].dtype == "object":
            # Convertir variables categóricas a dummies
            X = pd.get_dummies(X, columns=[col], drop_first=True)
    
    # Asegurarse de que todos los datos sean float
    X = X.astype(float)

    logging.info(f"Variables separadas. Features: {X.shape[1]} columnas, Target: {y.name}")

    # Retornar variables predictoras y objetivo
    return X, y

# ------------------------------------------------------------
# Función para dividir el dataset en conjuntos de entrenamiento y prueba
# ------------------------------------------------------------
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
    
    Lanza:
    - ValueError si X e y tienen diferente longitud
    """
    # Comprobar que X e y tengan la misma longitud
    if len(X) != len(y):
        raise ValueError(f"X e y deben tener la misma longitud. X tiene {len(X)} filas, y tiene {len(y)} filas.")
    
    # Dividir el dataset, estratificando según la variable objetivo
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logging.info(f"Datos divididos. Train: {X_train.shape[0]} filas, Test: {X_test.shape[0]} filas")
    
    return X_train, X_test, y_train, y_test

# ------------------------------------------------------------
# Función para escalar los datos con StandardScaler
# ------------------------------------------------------------
def escalar_datos(X_train, X_test):
    """
    Escala los datos con StandardScaler (media 0, varianza 1) y devuelve DataFrames.
    
    Parámetros:
    - X_train (pd.DataFrame): conjunto de entrenamiento
    - X_test (pd.DataFrame): conjunto de prueba
    
    Retorna:
    - X_train_scaled (pd.DataFrame): datos de entrenamiento escalados
    - X_test_scaled (pd.DataFrame): datos de prueba escalados
    - scaler (StandardScaler): objeto scaler, útil para aplicar la misma transformación a nuevos datos
    """
    # Crear objeto StandardScaler
    scaler = StandardScaler()
    
    # Ajustar scaler al conjunto de entrenamiento y transformar los datos
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),       # Escalar datos
        columns=X_train.columns,             # Mantener nombres de columnas
        index=X_train.index                  # Mantener índices originales
    )
    
    # Transformar conjunto de prueba usando el mismo scaler
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    logging.info("Datos escalados usando StandardScaler (media 0, varianza 1)")

    # Retornar datasets escalados y el objeto scaler
    return X_train_scaled, X_test_scaled, scaler
