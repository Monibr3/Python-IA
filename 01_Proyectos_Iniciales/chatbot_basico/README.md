# Chatbot con Memoria e IA (Hugging Face)

Este proyecto implementa un **chatbot en Python** que utiliza un **modelo de lenguaje preentrenado** de Hugging Face (BlenderBot o DialoGPT) para generar respuestas en inglés y **guarda las conversaciones en un archivo JSON** para poder consultarlas, borrarlas o mantener un historial.

## Características

- **Guardado de historial**: Todas las conversaciones se almacenan en `conversaciones.json`.
- **Recuperación de conversaciones previas**: Al iniciar, el chatbot puede mostrar el historial anterior.
- **Opciones de gestión de historial**: Permite borrar todo el historial o ver las conversaciones guardadas.
- **Filtrado de respuestas inapropiadas**: Detecta y evita que el bot responda con lenguaje ofensivo.
- **Generación de respuestas en inglés**: El modelo está entrenado para conversaciones naturales en inglés.

## Requisitos

- Python 3.8 o superior
- Paquetes de Python:
  - `transformers`
  - `torch`

Se recomienda crear un entorno virtual para instalar las dependencias.

