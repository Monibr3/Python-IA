# chatbot.py
# Este archivo maneja el chatbot usando Transformers y la gestión de conversaciones en JSON.

import json  # Librería para manejar archivos JSON (guardar y cargar conversaciones)
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
# Importamos el tokenizer y el modelo BlenderBot de Hugging Face Transformers

# Cargar el modelo BlenderBot (preentrenado, seguro y coherente)
modelo_nombre = "facebook/blenderbot-400M-distill"  # Nombre del modelo preentrenado de Hugging Face
tokenizer = BlenderbotTokenizer.from_pretrained(modelo_nombre)  
# Tokenizer: convierte texto en tensores que el modelo puede entender
modelo = BlenderbotForConditionalGeneration.from_pretrained(modelo_nombre)
# Modelo: BlenderBot listo para generar respuestas

# Archivo donde guardaremos las conversaciones
ARCHIVO_CONVERSACIONES = "conversaciones.json"

# Función para cargar conversaciones anteriores desde un archivo JSON
def cargar_conversaciones():
    try:
        # Intentamos abrir el archivo y cargar la lista de conversaciones
        with open(ARCHIVO_CONVERSACIONES, "r", encoding="utf-8") as f:
            return json.load(f)  # Devolver la lista de conversaciones
    except (FileNotFoundError, json.JSONDecodeError):
        # Si el archivo no existe o está vacío/corrupto, devolvemos una lista vacía
        return []

# Función para guardar conversaciones en JSON
def guardar_conversaciones(conversaciones):
    # Abre el archivo en modo escritura y guarda la lista de conversaciones
    with open(ARCHIVO_CONVERSACIONES, "w", encoding="utf-8") as f:
        json.dump(conversaciones, f, ensure_ascii=False, indent=4)
        # ensure_ascii=False mantiene acentos y caracteres especiales
        # indent=4 formatea el JSON de forma legible

# Función para filtrar respuestas inapropiadas
def filtrar_respuesta(texto):
    palabras_prohibidas = ["f***", "n***", "ur mum"]  # Lista de palabras que no queremos en las respuestas
    for palabra in palabras_prohibidas:
        if palabra.lower() in texto.lower():  # Comprobamos ignorando mayúsculas/minúsculas
            return "Lo siento, no puedo responder eso."  # Respuesta segura si hay palabra prohibida
    return texto  # Si no hay palabras prohibidas, devolvemos la respuesta tal cual

# Función principal del chat
def iniciar_chat():
    conversaciones = cargar_conversaciones()  # Cargamos historial previo al iniciar el chat

    # Mensajes de ayuda al usuario
    print("Escribe 'salir' para terminar la conversación.")
    print("Escribe 'borrar' para eliminar el historial anterior.")
    print("Escribe 'ver' para ver el historial anterior.\n")

    while True:
        entrada = input("Tú: ").strip()  # Leemos entrada del usuario y eliminamos espacios extra

        # Opción de salir del chat
        if entrada.lower() == "salir":
            opcion_guardar = input("¿Quieres guardar esta conversación? (s/n): ").strip().lower()
            if opcion_guardar == "s":
                guardar_conversaciones(conversaciones)  # Guardamos la conversación
                print("Conversación guardada.")
            else:
                print("Conversación descartada.")  # No se guarda la conversación
            break  # Salimos del bucle y terminamos el chat

        # Opción de borrar historial anterior
        if entrada.lower() == "borrar":
            conversaciones = []  # Reiniciamos la lista de conversaciones
            guardar_conversaciones(conversaciones)  # Sobrescribimos el archivo JSON
            print("Historial eliminado.")
            continue  # Volvemos al inicio del bucle

        # Opción de ver historial anterior
        if entrada.lower() == "ver":
            if conversaciones:
                print("\nHistorial de conversaciones:")
                for i, c in enumerate(conversaciones, 1):
                    print(f"{i}. Tú: {c['usuario']} → Bot: {c['bot']}")
                print("")  # Línea en blanco para separar
            else:
                print("No hay historial previo.\n")
            continue  # Volvemos al inicio del bucle

        # Tokenizar la entrada y generar respuesta usando BlenderBot
        entradas_ids = tokenizer(entrada, return_tensors="pt")
        # Convertimos la entrada del usuario a tensores PyTorch (formato que el modelo entiende)
        salida_ids = modelo.generate(**entradas_ids)
        # Generamos la respuesta del modelo
        respuesta = tokenizer.decode(salida_ids[0], skip_special_tokens=True)
        # Convertimos los IDs de salida de nuevo a texto, eliminando tokens especiales

        # Filtrar respuestas inapropiadas
        respuesta = filtrar_respuesta(respuesta)

        print(f"Bot: {respuesta}")  # Mostramos la respuesta al usuario

        # Guardar en la conversación actual
        conversaciones.append({"usuario": entrada, "bot": respuesta})
        # Cada interacción se guarda como un diccionario con 'usuario' y 'bot'
