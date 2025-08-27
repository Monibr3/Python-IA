# Importamos matplotlib para generar gráficos
import matplotlib.pyplot as plt

# Importamos seaborn para gráficos estadísticos más bonitos y sencillos
import seaborn as sns

# Importamos métricas de evaluación de scikit-learn
from sklearn.metrics import (
    accuracy_score,          # Exactitud del modelo
    precision_score,         # Precisión: proporción de predicciones positivas correctas
    recall_score,            # Recall o sensibilidad: proporción de positivos correctamente identificados
    f1_score,                # F1-score: media armónica de precisión y recall
    confusion_matrix,        # Matriz de confusión
    classification_report,   # Reporte completo de métricas por clase
    roc_curve,               # Curva ROC
    auc                      # Área bajo la curva ROC
)

def evaluar_modelo(y_test, y_pred):
    """
    Calcula métricas de evaluación básicas: accuracy, precision, recall, f1.
    
    Parámetros:
    - y_test (array-like): valores reales
    - y_pred (array-like): predicciones del modelo
    
    Retorna:
    - dict con métricas calculadas
    """
    
    # Calculamos cada métrica usando las funciones de sklearn
    resultados = {
        "accuracy": accuracy_score(y_test, y_pred),  # Exactitud
        "precision": precision_score(y_test, y_pred),  # Precisión
        "recall": recall_score(y_test, y_pred),  # Sensibilidad / Recall
        "f1_score": f1_score(y_test, y_pred),  # F1-score (media armónica de precision y recall)
    }
    
    # Devolvemos un diccionario con todas las métricas
    return resultados


def mostrar_matriz_confusion(y_test, y_pred):
    """
    Dibuja la matriz de confusión con seaborn.
    """
    
    # Calculamos la matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    
    # Configuramos el tamaño de la figura
    plt.figure(figsize=(6,4))
    
    # Dibujamos la matriz con anotaciones y colores
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=["No Diabetes", "Diabetes"], 
        yticklabels=["No Diabetes", "Diabetes"]
    )
    
    # Etiquetas y título
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title("Matriz de Confusión")
    
    # Mostramos la figura
    plt.show()


def informe_clasificacion(y_test, y_pred):
    """
    Imprime el infrome de clasificación detallado.
    """
    
    # Mensaje de cabecera
    print("Informe de Clasificación:")
    
    # Imprimimos el informe de clasificación usando sklearn
    print(classification_report(
        y_test, 
        y_pred, 
        target_names=["No Diabetes", "Diabetes"]
    ))


def curva_roc(y_test, y_proba):
    """
    Gráfica de la curva ROC y cálculo del AUC.
    
    Parámetros:
    - y_test (array-like): valores reales
    - y_proba (array-like): probabilidades estimadas por el modelo
    
    Retorna:
    - roc_auc (float): área bajo la curva ROC
    """
    
    # Calculamos los puntos de la curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    
    # Calculamos el área bajo la curva (AUC)
    roc_auc = auc(fpr, tpr)

    # Configuramos el tamaño de la figura
    plt.figure(figsize=(6,4))
    
    # Dibujamos la curva ROC
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")  # Línea diagonal de referencia
    
    # Etiquetas y título
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Curva ROC")
    plt.legend(loc="lower right")
    
    # Mostramos la figura
    plt.show()

    # Devolvemos el AUC
    return roc_auc
