# main.py
# Este archivo es el punto de entrada del chatbot.
# Importa la función principal iniciar_chat desde chatbot.py y la ejecuta.

from chatbot import iniciar_chat  
# Importamos la función iniciar_chat definida en chatbot.py
# Esta función maneja todo el flujo del chat: entrada del usuario, generación de respuestas,
# historial de conversaciones, opciones de ver/borrar/conservar historial, etc.

# Este bloque asegura que el script se ejecute solo si se llama directamente,
# y no si se importa como módulo en otro script.
if __name__ == "__main__":
    iniciar_chat()  
    # Llamamos a la función iniciar_chat para iniciar la interacción con el usuario
