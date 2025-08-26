# Proyecto: Segmentación de Clientes de E-commerce

## Descripción

Este proyecto realiza una **segmentación de clientes** de un e-commerce utilizando **aprendizaje no supervisado**, específicamente **clustering con KMeans**.  
El objetivo es identificar grupos de clientes con comportamientos de compra similares, con el fin de mejorar estrategias de marketing, fidelización y promociones.

---

## Objetivos

1. Preparar y limpiar los datos de clientes y transacciones.  
2. Calcular las métricas **RFM** (Recency, Frequency, Monetary) y métricas adicionales:
   - `Total_items`: número total de ítems comprados por cliente.  
   - `Max_purchase`: valor de la compra más alta.  
   - `Avg_ticket`: ticket promedio por pedido.  
   - `Avg_items_per_order`: promedio de ítems por pedido.  
3. Escalar las variables para clustering.  
4. Aplicar **KMeans** para agrupar clientes en clusters según su comportamiento de compra.  
5. Visualizar y analizar los clusters para interpretar los segmentos y proponer acciones de negocio.

---

## Datos

Se utilizan los datasets del e-commerce **Olist**:

- `olist_customers_dataset.csv`: información de clientes.  
- `olist_orders_dataset.csv`: información de pedidos.  
- `olist_order_items_dataset.csv`: información de ítems de cada pedido.

---

## Metodología

1. **Carga y limpieza de datos**  
   - Se unifican las tablas de clientes, pedidos y transacciones.  
   - Se revisan columnas, tipos de datos y duplicados.  

2. **Cálculo de métricas RFM y adicionales**  
   - **Recency**: días desde la última compra hasta la fecha de referencia.  
   - **Frequency**: número de pedidos realizados por cliente.  
   - **Monetary**: gasto total del cliente.  
   - Métricas adicionales: `Total_items`, `Max_purchase`, `Avg_ticket`, `Avg_items_per_order`.  

3. **Escalado de variables**  
   - Se utiliza `StandardScaler` para normalizar las métricas y evitar que una variable domine el clustering.  

4. **Clustering con KMeans**  
   - Se determina el número óptimo de clusters usando el **método del codo** y, opcionalmente, el **silhouette score**.  
   - Se aplica KMeans y se asigna a cada cliente su cluster correspondiente.  

5. **Visualización y análisis de clusters**  
   - Scatter plots de `Recency` vs `Monetary` por cluster.  
   - Histogramas de distribución de métricas por cluster.  
   - Radar charts para comparar perfiles promedio de cada cluster.  

---

## Resultados

- Los clientes se agruparon en 5 clusters principales, con diferencias en recencia, gasto total y cantidad de ítems.  
- Segmentos identificados:
  - Clientes recientes con bajo gasto.  
  - Clientes antiguos con bajo gasto.  
  - Clientes moderadamente activos con gasto medio.  
  - Clientes de alto gasto en pocas compras.  
  - Clientes excepcionales con gasto muy alto.  
- Esta segmentación permite diseñar estrategias de marketing personalizadas y priorizar clientes de alto valor.

---

## Tecnologías y librerías

- Python 3.x  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn (`StandardScaler`, `KMeans`)  

---

## Instrucciones para ejecutar

1. Clonar este repositorio.  
2. Instalar dependencias (recomendado crear un entorno virtual):
