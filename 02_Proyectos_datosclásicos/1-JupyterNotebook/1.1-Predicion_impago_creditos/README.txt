Predicción de riesgo de impago de créditos
Este proyecto utiliza Python y Aprendizaje Supervisado para predecir si un cliente incumplirá con el pago de su crédito el mes siguiente. Está desarrollado en Jupyter Notebook y se enfoca en un dataset de clientes con información financiera y personal.
________________________________________
1. Objetivo
El objetivo es identificar clientes con alto riesgo de impago, ayudando a instituciones financieras a reducir pérdidas y tomar decisiones más informadas sobre la concesión de créditos.
•	Variable objetivo: default_payment_next_month
o	0 → No impago
o	1 → Impago
•	Características (features): edad, sexo, límite de crédito, historial de pagos, facturas y pagos anteriores, entre otras.
________________________________________
2. Preprocesamiento
•	Lectura del dataset CSV y renombrado de columnas para facilitar el trabajo.
•	Eliminación de columnas irrelevantes (id).
•	Escalado de variables con StandardScaler para normalizar los valores y mejorar el rendimiento del modelo.
•	División del dataset en 80% entrenamiento y 20% prueba, manteniendo la proporción de impagos (stratify=y).
________________________________________
3. Modelo elegido
Se utilizó Random Forest con la opción class_weight='balanced':
•	Ensamble de múltiples árboles de decisión.
•	Capaz de capturar relaciones complejas no lineales entre las variables.
•	Balancea automáticamente la importancia de las clases minoritarias (impagos) en el entrenamiento.
________________________________________
4. Resultados
Métricas principales en el conjunto de prueba:
Métrica	Clase 0 (no impago)	Clase 1 (impago)
Precision	0.87	0.52
Recall	0.85	0.56
F1-score	0.86	0.54
•	Accuracy global: 0.789 (~79%)
•	El modelo detecta más de la mitad de los impagos reales y mantiene una buena exactitud global.
Este equilibrio hace que el modelo sea robusto y práctico para un entorno real de toma de decisiones financieras.
________________________________________
5. Uso
•	Entrenar el modelo con el dataset preparado.
•	Escalar los datos de los nuevos clientes.
•	Predecir si un cliente es probable que incumpla el pago del próximo mes.
Esto permite a los analistas y bancos identificar clientes de alto riesgo y tomar decisiones más informadas.

