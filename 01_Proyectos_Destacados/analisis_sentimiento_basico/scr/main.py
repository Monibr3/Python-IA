# main.py
import os
import joblib
from modelo import preprocesado  # Importamos la función de limpieza de tweets

# =========================
# 1. CARGAR MODELO Y VECTORIZADOR DESDE LA CARPETA 'modelos'
# =========================

try:
    # Definimos la ruta de la carpeta donde guardamos los modelos
    ruta_modelos = os.path.join(os.path.dirname(__file__), 'modelos')

    # Cargamos el modelo entrenado
    modelo = joblib.load(os.path.join(ruta_modelos, 'sentiment_model.pkl'))

    # Cargamos el vectorizador
    vectorizador = joblib.load(os.path.join(ruta_modelos, 'vectorizador.pkl'))

    print("Modelo y vectorizador cargados correctamente.")

except FileNotFoundError as e:
    # Capturamos el error si alguno de los archivos no se encuentra
    print(f"No se encontró el archivo: {e}")
    exit(1)

except Exception as e:
    # Capturamos cualquier otro error inesperado al cargar los archivos
    print(f"Ocurrió un error al cargar modelo/vectorizador: {e}")
    exit(1)

# =========================
# 2. FUNCIÓN DE PREDICCIÓN
# =========================
def predecir_sentimiento(texto):
    """
    Recibe un tweet nuevo, lo limpia y devuelve si su sentimiento es positivo o negativo.
    Pasos:
    1. Limpieza del texto usando la función preprocesado().
    2. Transformación del texto en vector numérico mediante el vectorizador cargado.
    3. Predicción usando el modelo cargado.
    """
    texto_limpio = preprocesado(texto)
    vector = vectorizador.transform([texto_limpio])
    return modelo.predict(vector)[0]

# =========================
# 3. INTERFAZ POR PANTALLA
# =========================
if __name__ == "__main__":
    print("¡Bienvenido! Introduce tweets para analizar su sentimiento. Escribe 'salir' para terminar.")
    while True:
        tweet_usuario = input("Tweet: ")
        if tweet_usuario.lower() == "salir":
            print("¡Hasta luego!")
            break
        sentimiento = predecir_sentimiento(tweet_usuario)
        print(f"El sentimiento del tweet es: {sentimiento}\n")
