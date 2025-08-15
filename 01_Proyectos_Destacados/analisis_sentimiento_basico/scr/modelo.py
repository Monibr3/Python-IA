# =========================
# Importación de librerías
# =========================

import os                          # Para manejar rutas, verificar si archivos existen, etc.
import pandas as pd                # Para manipular datos en forma de tablas (DataFrames)
import re                          # Para usar expresiones regulares y limpiar texto
from sklearn.feature_extraction.text import TfidfVectorizer  # Para convertir texto a vectores numéricos
from sklearn.model_selection import train_test_split        # Para dividir los datos en entrenamiento y prueba
from sklearn.linear_model import LogisticRegression         # Modelo de clasificación lineal
from sklearn.metrics import accuracy_score                  # Para medir la precisión del modelo
import joblib                      # Para guardar y cargar modelos y objetos de Python
import sys                         # Para poder salir del script si ocurre un error crítico

# =========================
# 1. CARGAR LOS DATOS CON MANEJO DE EXCEPCIONES
# =========================

# Ruta del dataset que vamos a usar
ruta_csv = r"C:\Archivos de trabajo\Python-IA\analisis_sentimiento_basico\Data\training.1600000.processed.noemoticon.csv"

try:
    # Verificamos si el archivo existe antes de intentar leerlo
    if not os.path.exists(ruta_csv):
        # Si no existe, lanzamos una excepción FileNotFoundError con un mensaje explicativo
        raise FileNotFoundError(f"No se encontró el archivo: {ruta_csv}")

    # Intentamos leer el archivo CSV
    # encoding="latin-1" se usa porque algunos tweets antiguos tienen caracteres especiales
    # header=None porque el CSV no tiene encabezado
    datos = pd.read_csv(ruta_csv, encoding="latin-1", header=None)

except FileNotFoundError as e:
    # Capturamos el error si el archivo no existe
    print("Error:", e)
    sys.exit(1)  # Salimos del programa porque sin datos no podemos continuar

except pd.errors.ParserError as e:
    # Capturamos errores si el CSV está corrupto o mal formateado
    print("Error al leer el CSV. Posiblemente está corrupto o mal formateado.")
    print(e)
    sys.exit(1)

except Exception as e:
    # Capturamos cualquier otro error inesperado
    print("Ocurrió un error inesperado al leer el archivo CSV.")
    print(e)
    sys.exit(1)

# =========================
# 2. SELECCIÓN DE COLUMNAS Y RENOMBRADO
# =========================

# El dataset original tiene muchas columnas, pero solo necesitamos:
# - columna 0: sentimiento (0 = negativo, 4 = positivo)
# - columna 5: texto del tweet
df = datos[[0, 5]].copy()  # Hacemos una copia para evitar advertencias de pandas

# Renombramos columnas a nombres más descriptivos
df.columns = ["sentimiento", "tweet"]

# Convertimos las etiquetas numéricas a texto para mayor legibilidad
# 0 -> "negative" (negativo)
# 4 -> "positive" (positivo)
df['sentimiento'] = df['sentimiento'].map({0: "negativo", 4: "positivo"})

# Para acelerar el entrenamiento, seleccionamos aleatoriamente 50.000 registros
# random_state asegura que la selección sea reproducible
df = df.sample(50000, random_state=42)

# Mostramos las primeras filas para verificar que todo esté correcto
print("Primeras filas del dataset:")
print(df.head())
print("\nNúmero total de registros seleccionados:", len(df))

# =========================
# 3. PREPROCESAMIENTO DEL TEXTO
# =========================

def preprocesado(texto):
    """
    Función que limpia el texto de cada tweet antes de vectorizarlo.
    Pasos:
    1. Verifica que el dato sea una cadena; si no lo es, devuelve un string vacío.
    2. Convierte todo el texto a minúsculas.
    3. Elimina URLs (http, https, www).
    4. Elimina menciones (@usuario).
    5. Elimina hashtags (#ejemplo).
    6. Elimina caracteres no alfabéticos (números, signos de puntuación, emojis, etc.).
    """
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()  # Convertimos todo a minúsculas
    texto = re.sub(r"http\S+|www\S+|https\S+", '', texto)  # Eliminamos URLs
    texto = re.sub(r'@\w+', '', texto)                     # Eliminamos menciones
    texto = re.sub(r'#\w+', '', texto)                     # Eliminamos hashtags
    texto = re.sub(r'[^a-zA-Z\s]', '', texto)             # Eliminamos caracteres no alfabéticos
    return texto

# Aplicamos la función de preprocesado a toda la columna 'tweet'
df['tweet'] = df['tweet'].apply(preprocesado)

# Mostramos los primeros tweets procesados para comprobar resultados
print("\nPrimeros tweets después de preprocesar:")
print(df['tweet'].head())

# =========================
# 4. TRANSFORMACIÓN DE TEXTO A VECTORES
# =========================

# Los modelos de machine learning no entienden texto directamente, necesitan números
# TfidfVectorizer convierte cada tweet en un vector numérico basado en la frecuencia de palabras
# max_features=5000 limita el vocabulario a las 5000 palabras más relevantes
vectorizador = TfidfVectorizer(max_features=5000)

print("\nTransformando tweets en vectores numéricos...")
X = vectorizador.fit_transform(df['tweet'])  # Convertimos los tweets en vectores
y = df['sentimiento']                        # Las etiquetas (positivo/negativo)

# =========================
# 5. DIVISIÓN DE DATOS EN ENTRENAMIENTO Y PRUEBA
# =========================

# Dividimos los datos en:
# - 80% para entrenamiento (aprendizaje del modelo)
# - 20% para prueba (evaluación del modelo)
print("Dividiendo datos en entrenamiento y prueba...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 6. ENTRENAMIENTO DEL MODELO
# =========================

# Logistic Regression es un modelo simple y efectivo para clasificación binaria de texto
modelo = LogisticRegression()

print("Entrenando el modelo...")
# Entrenamos el modelo con los datos de entrenamiento
modelo.fit(X_train, y_train)

# =========================
# 7. EVALUACIÓN DEL MODELO
# =========================

print("Evaluando el modelo...")
# Predecimos los sentimientos de los tweets de prueba
y_pred = modelo.predict(X_test)

# Calculamos la precisión (accuracy) que indica el porcentaje de predicciones correctas
precision = accuracy_score(y_test, y_pred)
print("Precisión del modelo:", precision)

# =========================
# 8. GUARDAR MODELO Y VECTORIZADOR
# =========================
# =========================
# 8. GUARDAR MODELO Y VECTORIZADOR EN UNA CARPETA DEDICADA
# =========================

try:
    # Definimos la carpeta donde queremos guardar los modelos
    # os.path.dirname(__file__) nos da la ruta del script actual
    ruta_modelos = os.path.join(os.path.dirname(__file__), 'modelos')
    
    # Creamos la carpeta si no existe. exist_ok=True evita errores si ya existe
    os.makedirs(ruta_modelos, exist_ok=True)

    # Guardamos el modelo entrenado dentro de la carpeta 'modelos'
    joblib.dump(modelo, os.path.join(ruta_modelos, 'sentiment_model.pkl'))

    # Guardamos el vectorizador dentro de la misma carpeta
    joblib.dump(vectorizador, os.path.join(ruta_modelos, 'vectorizador.pkl'))

    print("Modelo y vectorizador guardados correctamente en la carpeta 'modelos'.")

except Exception as e:
    # Capturamos cualquier error al intentar guardar los archivos
    print(f"Error al guardar modelo/vectorizador: {e}")
