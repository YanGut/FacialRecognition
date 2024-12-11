import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Constroi uma matriz de confusão.
    
    Args:
        y_true (np.ndarray): Vetor de rótulos reais.
        y_pred (np.ndarray): Vetor de predições.
    
    Returns:
        np.ndarray: Matriz de confusão.
    """

    true_positive = np.sum((y_pred == 1) & (y_true == 1))
    true_negative = np.sum((y_pred == -1) & (y_true == -1))
    false_positive = np.sum((y_pred == 1) & (y_true == -1))
    false_negative = np.sum((y_pred == -1) & (y_true == 1))
    
    return np.array([
        [true_positive, false_positive],
        [false_negative, true_negative]
    ])

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
    
    sns.heatmap(confusion_matrix_x1, annot=True, fmt="d", cmap="Blues", xticklabels=["Pred -1", "Pred 1"], yticklabels=["True -1", "True 1"], ax=axes[0])
    axes[0].set_title(f"Melhor {model_name}")
    axes[0].set_xlabel("Previsão")
    axes[0].set_ylabel("Valor Real")
    
    # Pior modelo
    sns.heatmap(confusion_matrix_x2, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Pred -1", "Pred 1"], yticklabels=["True -1", "True 1"], ax=axes[1])
    axes[1].set_title(f"Pior {model_name}")
    axes[1].set_xlabel("Previsão")
    axes[1].set_ylabel("Valor Real")
    
    plt.tight_layout()
    plt.show()