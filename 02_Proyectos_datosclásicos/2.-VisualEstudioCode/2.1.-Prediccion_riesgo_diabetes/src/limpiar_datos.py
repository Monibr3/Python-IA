import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def revisar_nulos(df):
    """
    Revisa si hay valores nulos y los muestra por pantalla.
    """
    
    # Calcula cuántos valores nulos hay en cada columna del DataFrame.
    # df.isnull() devuelve True donde hay un nulo (NaN), y .sum() cuenta los True por columna.
    nulos = df.isnull().sum()

    # Imprime un encabezado para indicar que a continuación se muestran los valores nulos.
    print("Valores nulos por columna:")

    # Si existe al menos un valor nulo en todo el DataFrame (nulos.sum() > 0),
    # se imprimen únicamente las columnas que tienen nulos (nulos > 0).
    # En caso contrario, se muestra el mensaje "No hay valores nulos".
    print(nulos[nulos > 0] if nulos.sum() > 0 else "No hay valores nulos")

    # Devuelve la Serie con el número de nulos en cada columna,
    # aunque en pantalla sólo se muestren las que tengan valores nulos.
    return nulos

def reemplazar_nulos(df, columnas=None, metodo="mediana"):
    """
    Reemplaza valores nulos en columnas específicas o todas.
    
    Parámetros:
    - df (pd.DataFrame)
    - columnas (list de str) -> columnas a limpiar, None = todas
    - metodo (str) -> 'media' o 'mediana'
    """
    
    # Si no se han indicado columnas, se seleccionan todas las columnas del DataFrame
    if columnas is None:
        columnas = df.columns
    
    # Recorremos cada columna indicada
    for col in columnas:
        # Comprobamos si la columna tiene algún valor nulo
        if df[col].isnull().sum() > 0:
            # Calculamos el valor que se usará para reemplazar los nulos
            if metodo == "media":
                valor = df[col].mean()   # Se usa la media de la columna
            else:
                valor = df[col].median() # Se usa la mediana de la columna
            
            # Rellenamos los valores nulos con el valor calculado
            # inplace=True indica que se modifica la columna directamente en el DataFrame
            df[col].fillna(valor, inplace=True)
            
            # Mostramos un mensaje indicando qué columna se ha procesado, con qué método y valor
            print(f"Se reemplazaron valores nulos en {col} con {metodo}: {valor}")
    
    # Devolvemos el DataFrame con los valores nulos ya reemplazados
    return df

def detectar_outliers(df, columnas=None, limite=1.5, graficar=False):
    """
    Detecta outliers en un DataFrame usando el método del rango intercuartílico (IQR).
    
    Parámetros:
    - df: DataFrame de entrada
    - columnas: lista de columnas a revisar (si es None, se usan todas las numéricas)
    - limite: multiplicador del IQR (1.5 es el valor estándar)
    - graficar: si es True, muestra boxplots de las columnas
    """
    
    # Si no se especifican columnas, se toman todas las columnas numéricas del DataFrame
    if columnas is None:
        columnas = df.select_dtypes(include="number").columns

    resultados = {}  # Diccionario para almacenar el número de outliers por columna
    
    # Iteramos por cada columna indicada
    for col in columnas:
        # Calculamos el primer cuartil (Q1, 25%) y el tercer cuartil (Q3, 75%)
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        
        # Calculamos el rango intercuartílico (IQR)
        IQR = Q3 - Q1
        
        # Definimos los límites para detectar outliers
        limite_inferior = Q1 - limite * IQR
        limite_superior = Q3 + limite * IQR
        
        # Filtramos las filas que están fuera de los límites (son outliers)
        outliers = df[(df[col] < limite_inferior) | (df[col] > limite_superior)]
        
        # Guardamos el número de outliers encontrados en el diccionario
        resultados[col] = len(outliers)
        
        # Si se indica graficar, mostramos un boxplot de la columna
        if graficar:
            sns.boxplot(x=df[col])
            plt.title(f"Boxplot de {col}")
            plt.show()
        
        # Mostramos en pantalla el número de outliers detectados en la columna
        print(f"{col}: {len(outliers)} outliers detectados")
    
    # Devolvemos el diccionario con el número de outliers por columna
    return resultados


def limpiar_dataset(df, columnas_nulos=None, metodo="mediana", columnas_outliers=None):
    """
    Limpia el dataset:
    - Reemplaza valores nulos
    - Detecta y opcionalmente elimina outliers
    
    Retorna:
    - df limpio
    """
    
    # Aviso en pantalla indicando que se va a revisar la presencia de valores nulos
    print("Revisando nulos...")
    
    # Llamada a la función que revisa los nulos y los muestra por pantalla
    revisar_nulos(df)
    
    # Si se han indicado columnas donde reemplazar nulos, se aplicará la función correspondiente
    if columnas_nulos:
        df = reemplazar_nulos(df, columnas=columnas_nulos, metodo=metodo)
    
    # Si se han indicado columnas para detectar outliers, se revisan una por una
    if columnas_outliers:
        for col in columnas_outliers:
            # Detecta outliers y muestra un boxplot para cada columna
            detectar_outliers(df, col, graficar=True)
    
    # Mensaje final indicando que el dataset ha sido revisado
    print("Dataset revisado y limpio")
    
    # Devuelve el DataFrame ya limpio
    return df

