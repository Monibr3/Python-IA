import pandas as pd

def cargar_datos(ruta_csv):
    """
    Carga el dataset de diabetes desde un archivo CSV y realiza comprobaciones básicas.
    
    Parámetros:
    ruta_csv (str): Ruta del archivo CSV
    
    Retorna:
    pd.DataFrame: DataFrame con los datos cargados
    """
    try:
        df = pd.read_csv(ruta_csv)
        print(f"Datos cargados correctamente: {df.shape[0]} filas, {df.shape[1]} columnas")
        
        # Comprobaciones básicas
        if df.isnull().sum().sum() > 0:
            print(" Atención: hay valores nulos en el dataset")
        else:
            print("No hay valores nulos")
        
        return df
    
    except FileNotFoundError:
        print(f"Error: archivo no encontrado en la ruta {ruta_csv}")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: archivo vacío en {ruta_csv}")
        return None
    except Exception as e:
        print(f"Error al cargar el dataset: {e}")
        return None
