import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, number_classes: int) -> np.ndarray:
    """
    Calcula a matriz de confusão.
    
    Args:
        y_true (np.ndarray): Vetor de rótulos reais.
        y_pred (np.ndarray): Vetor de rótulos preditos.
        number_classes (int): Número de classes.
    
    Returns:
        np.ndarray: Matriz de confusão.
    """
    
    confusion_matrix = np.zeros((number_classes, number_classes), dtype=int)
    
    for true_label, predicted_label in zip(y_true, y_pred):
        confusion_matrix[true_label, predicted_label] += 1
    return confusion_matrix

def plot_confusion_matrix(
    confusion_matrix_x1: np.ndarray,
    confusion_matrix_x2: np.ndarray,
    model_name: str
) -> None:
    """
    Plota duas matrizes de confusão.
    
    Args:
        confusion_matrix_x1 (np.ndarray): Matriz de confusão do melhor modelo.
        confusion_matrix_x2 (np.ndarray): Matriz de confusão do pior modelo.
        model_name (str): Nome do modelo.
    
    Returns:
        None
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    labels = ["an2i", "at33", "bol", "bpm", "ch4f", "cheyer", "choon", "danieln", "glickman", "karyadi", "kawamura", "kk49", "megak", "mitchell", "night", "phoebe", "saavik", "steffi", "sz24", "tammo"]
    
    sns.heatmap(confusion_matrix_x1, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title(f"Melhor {model_name}")
    axes[0].set_xlabel("Previsão")
    axes[0].set_ylabel("Valor Real")
    
    # Pior modelo
    sns.heatmap(confusion_matrix_x2, annot=True, fmt="d", cmap="Blues", 
                xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_title(f"Pior {model_name}")
    axes[1].set_xlabel("Previsão")
    axes[1].set_ylabel("Valor Real")
    
    plt.tight_layout()
    plt.show()