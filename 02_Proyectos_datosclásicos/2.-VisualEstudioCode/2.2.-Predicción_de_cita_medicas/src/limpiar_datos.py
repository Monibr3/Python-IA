import pandas as pd      # Manejo de DataFrames y CSV
import seaborn as sns    # Gráficos estadísticos
import matplotlib.pyplot as plt  # Gráficos generales
import logging           # Para mostrar mensajes profesionales en lugar de print
from typing import Optional

# Configuración básica del logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def revisar_nulos(df: pd.DataFrame) -> pd.Series:
    """
    Revisa la existencia de valores nulos en el DataFrame.
    Retorna una Serie con el conteo de nulos por columna.
    """
    # Contar valores nulos en cada columna
    nulos = df.isnull().sum()
    
    # Mostrar advertencia solo si hay valores nulos
    if nulos.sum() > 0:
        logging.warning(f"Valores nulos por columna:\n{nulos[nulos > 0]}")
    else:
        logging.info("No hay valores nulos")
    
    # Retornar Serie con número de nulos por columna
    return nulos


def reemplazar_nulos(
    df: pd.DataFrame,
    columnas: Optional[list[str]] = None,
    metodo: str = "mediana"
) -> pd.DataFrame:
    """
    Reemplaza valores nulos solo en columnas numéricas.
    """
    # Si no se indican columnas, usar todas las columnas numéricas
    if columnas is None:
        columnas = df.select_dtypes(include="number").columns

    # Iterar por cada columna seleccionada
    for col in columnas:
        # Solo reemplazar si la columna tiene valores nulos
        if df[col].isnull().sum() > 0:
            # Calcular el valor a usar: media o mediana
            valor = df[col].mean() if metodo == "media" else df[col].median()
            
            # Reemplazar valores nulos con el valor calculado
            df[col].fillna(valor, inplace=True)
            
            # Informar del reemplazo realizado
            logging.info(f"Valores nulos en '{col}' reemplazados por {metodo}: {valor}")

    # Retornar DataFrame modificado
    return df


def detectar_outliers(
    df: pd.DataFrame,
    columnas: Optional[list[str]] = None,
    limite: float = 1.5,
    graficar: bool = False
) -> dict[str, int]:
    """
    Detecta outliers solo en columnas numéricas usando el rango intercuartílico (IQR).
    Retorna un diccionario con el número de outliers por columna.
    """
    # Si no se especifican columnas, usar todas las columnas numéricas
    if columnas is None:
        columnas = df.select_dtypes(include="number").columns

    resultados = {}  # Diccionario para guardar número de outliers por columna

    # Iterar por cada columna seleccionada
    for col in columnas:
        # Calcular Q1 (25%) y Q3 (75%) para el rango intercuartílico
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1  # Rango intercuartílico

        # Calcular límites inferior y superior para detectar outliers
        limite_inferior = Q1 - limite * IQR
        limite_superior = Q3 + limite * IQR

        # Filtrar filas que están fuera de los límites
        outliers = df[(df[col] < limite_inferior) | (df[col] > limite_superior)]

        # Guardar número de outliers detectados en el diccionario
        resultados[col] = len(outliers)

        # Mostrar boxplot si se solicita
        if graficar:
            sns.boxplot(x=df[col])
            plt.title(f"Boxplot de {col}")
            plt.show()

        # Informar cuántos outliers se detectaron
        if len(outliers) > 0:
            logging.warning(f"{col}: {len(outliers)} outliers detectados")
        else:
            logging.info(f"{col}: No se detectaron outliers")

    # Retornar diccionario con resultados
    return resultados


def limpiar_dataset(
    df: pd.DataFrame,
    columnas_nulos: Optional[list[str]] = None,
    metodo: str = "mediana",
    columnas_outliers: Optional[list[str]] = None
) -> pd.DataFrame:
    """
    Limpia el dataset de citas médicas reemplazando nulos en columnas numéricas
    y detectando outliers solo en columnas numéricas.
    """
    # Revisar valores nulos antes de reemplazar
    logging.info("Revisión de valores nulos...")
    revisar_nulos(df)

    # Reemplazo de nulos en columnas específicas si se indican
    if columnas_nulos:
        # Filtrar solo columnas numéricas
        num_cols = [c for c in columnas_nulos if df[c].dtype in ['int64', 'float64']]
        if num_cols:
            df = reemplazar_nulos(df, columnas=num_cols, metodo=metodo)

    # Detección de outliers en columnas específicas si se indican
    if columnas_outliers:
        # Filtrar solo columnas numéricas
        num_cols = [c for c in columnas_outliers if df[c].dtype in ['int64', 'float64']]
        for col in num_cols:
            detectar_outliers(df, [col], graficar=True)

    # Mensaje final indicando que el proceso terminó
    logging.info("Proceso de limpieza finalizado.")

    # Retornar DataFrame limpio
    return df
