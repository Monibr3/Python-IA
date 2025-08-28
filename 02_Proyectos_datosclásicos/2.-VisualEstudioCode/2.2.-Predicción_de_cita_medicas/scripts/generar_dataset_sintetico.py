"""
Script: generar_dataset_sintetico.py
------------------------------------
Objetivo: Generar un dataset sintético de citas médicas para proyectos de predicción de demanda
en centros de salud, útil para entrenamiento de modelos de machine learning y análisis exploratorio (EDA).

Contexto:
- Cada registro representa el número de citas diarias de una especialidad en un centro de salud.
- Se simulan variaciones realistas según día de la semana (menos citas en fines de semana).
- Este dataset permite probar pipelines de predicción y visualizaciones antes de acceder a datos reales.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# ----------------------------
# 1. Parámetros iniciales del dataset
# ----------------------------
start_date = datetime(2023, 1, 1)
end_date = datetime(2025, 1, 1)

num_centros = 5

# Especialidades más representativas de un centro de salud general
especialidades = ['Medicia', 'Pediatría', 'Farmacia', 'Enfermería', 'Odontología']

centros = [f'Centro_{i+1}' for i in range(num_centros)]

# ----------------------------
# 2. Generación del rango de fechas
# ----------------------------
date_range = pd.date_range(start=start_date, end=end_date, freq='D')

# ----------------------------
# 3. Generación del dataset sintético
# ----------------------------
data = []

for fecha in date_range:
    for centro in centros:
        for esp in especialidades:
            base_citas = random.randint(5, 20)
            
            if fecha.weekday() >= 5:  # Sábado y domingo
                num_citas = max(0, int(base_citas * random.uniform(0.3, 0.7)))
            else:  # Días laborables
                num_citas = int(base_citas * random.uniform(0.8, 1.2))
            
            data.append([fecha, centro, esp, num_citas])

# ----------------------------
# 4. Crear DataFrame
# ----------------------------
df_citas = pd.DataFrame(data, columns=['fecha_cita', 'centro_salud', 'especialidad', 'num_citas'])

# ----------------------------
# 5. Guardar dataset en CSV
# ----------------------------
df_citas.to_csv('data/citas_sinteticas.csv', index=False)

print("Dataset sintético generado: data/citas_sinteticas.csv")
print("Dimensiones del dataset:", df_citas.shape)
df_citas.head()
