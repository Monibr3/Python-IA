import pandas as pd  # Librería principal para manejo de datos en DataFrames
import logging

# Configuración básica del logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def cargar_datos(ruta_csv: str) -> pd.DataFrame | None:
    """
    Carga el dataset desde un archivo CSV y realiza comprobaciones básicas.

    Parámetros
    ----------
    ruta_csv : str
        Ruta al archivo CSV que contiene el dataset.

    Retorna
    -------
    pd.DataFrame | None
        - DataFrame con los datos cargados si la lectura fue exitosa.
        - None si ocurre un error (archivo no encontrado, vacío o lectura fallida).
    """

    try:
        # Intentar leer el archivo CSV en un DataFrame
        df = pd.read_csv(ruta_csv)

        # Mostrar dimensiones del dataset (número de filas y columnas)
        logging.info(f"Datos cargados correctamente: {df.shape[0]} filas, {df.shape[1]} columnas")
        
        # Verificar si existen valores nulos en todo el DataFrame
        nulos = df.isnull().sum()
        if nulos.sum() > 0:
            logging.warning(f"Atención: hay valores nulos en el dataset\nValores nulos por columna:\n{nulos}")
        else:
            logging.info("No hay valores nulos")

        # Retornar el DataFrame cargado
        return df

    except FileNotFoundError:
        # Manejo específico si el archivo no existe
        logging.error(f"Archivo no encontrado en la ruta {ruta_csv}")
        return None

    except pd.errors.EmptyDataError:
        # Manejo si el archivo está vacío
        logging.error(f"Archivo vacío en {ruta_csv}")
        return None

    except Exception as e:
        # Manejo de cualquier otro error inesperado
        logging.error(f"Error al cargar el dataset: {e}")
        return None
