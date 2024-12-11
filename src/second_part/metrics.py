import numpy as np
from typing import List, Tuple, Dict, Union
import pandas as pd
import matplotlib.pyplot as plt
from matrices import calculate_confusion_matrix

def calculate_metrics(predictions: np.ndarray, Y_test: np.ndarray, n_classes: int) -> Tuple[float, float, float, np.ndarray]:
    """
    Calcula as métricas de acurácia, sensibilidade e especificidade.
    
    Args:
        predictions (np.ndarray): Vetor de predições.
        Y_test (np.ndarray): Vetor de rótulos reais.
        n_classes (int): Número de classes.
    
    Returns:
        Tuple[float, float, float, np.ndarray]: Acurácia, Sensibilidade, Especificidade e Matriz de Confusão.
    """
    
    accuracy = np.mean(predictions == Y_test)
    
    confusion_matrix = calculate_confusion_matrix(Y_test, predictions, n_classes)
    
    sensitivity = []
    specificity = []
    
    for i in range(n_classes):
        true_positive = confusion_matrix[i, i]
        false_positive = confusion_matrix[:, i].sum() - true_positive
        false_negative = confusion_matrix[i, :].sum() - true_positive
        true_negative = np.sum(confusion_matrix) - (true_positive + false_positive + false_negative)
        
        sensitivity.append(true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0)
        specificity.append(true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0)
    
    average_sensitivity = np.mean(sensitivity)
    average_specificity = np.mean(specificity)
    
    return accuracy, average_sensitivity, average_specificity, confusion_matrix
    

def update_results(
    metrics: Tuple[float, float, float],
    results: List[Dict[str, Union[float, np.ndarray]]],
    predictions: np.ndarray,
    Y_test: np.ndarray,
    mse_history: List[float],
    n_classes: int
) -> None:
    """
    Atualiza os resultados do modelo.
    
    Args:
        metrics (Tuple[float, float, float]): Acurácia, Sensibilidade e Especificidade.
        results (List[Dict[str, Union[float, np.ndarray]]]): Lista de resultados.
        predictions (np.ndarray): Vetor de predições.
        Y_test (np.ndarray): Vetor de rótulos reais.
        mse_history (List[float]): Lista de erros médios quadráticos.
    
    Returns:
        None
    """
    
    accuracy, sensitivity, specificity, confusion_matrix = calculate_metrics(predictions, np.argmax(Y_test, axis=1), n_classes)

    metrics["accuracy"].append(accuracy)
    metrics["sensitivity"].append(sensitivity)
    metrics["specificity"].append(specificity)

    results.append({"accuracy": accuracy, "conf_matrix": confusion_matrix, "mse": mse_history})

def build_summary(metrics: Tuple[float, float, float]) -> pd.DataFrame:
    """
    Constroi um resumo das métricas.
    
    Args:
        metrics (Tuple[float, float, float]): Acurácia, Sensibilidade e Especificidade.
    
    Returns:
        pd.DataFrame: Resumo das métricas.
    """
    
    return pd.DataFrame({
        "Metric": ["Accuracy", "Sensitivity", "Specificity"],
        "Mean": [
            np.mean(metrics["accuracy"]),
            np.mean(metrics["sensitivity"]),
            np.mean(metrics["specificity"]),
        ],
        "StdDev": [
            np.std(metrics["accuracy"]),
            np.std(metrics["sensitivity"]),
            np.std(metrics["specificity"]),
        ],
        "Max": [
            np.max(metrics["accuracy"]),
            np.max(metrics["sensitivity"]),
            np.max(metrics["specificity"]),
        ],
        "Min": [
            np.min(metrics["accuracy"]),
            np.min(metrics["sensitivity"]),
            np.min(metrics["specificity"]),
        ],
    })

def plot_leaning_curves(best_case_mse: List[float], worst_case_mse: List[float], title: str) -> None:
    """
    Plota as curvas de aprendizado.
    
    Args:
        best_case_mse (List[float]): Lista de erros médios quadráticos do melhor caso.
        worst_case_mse (List[float]): Lista de erros médios quadráticos do pior caso.
        title (str): Título do gráfico.
    
    Returns:
        None
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    
    axs[0].plot(range(1, len(best_case_mse) + 1), best_case_mse, marker='o', linestyle='-', color='g')
    axs[0].set_title(f"{title} - Melhor Caso")
    axs[0].set_xlabel("Épocas")
    axs[0].set_ylabel("Erro Quadrático Médio (MSE)")
    axs[0].grid(True)

    # Pior caso
    axs[1].plot(range(1, len(worst_case_mse) + 1), worst_case_mse, marker='o', linestyle='-', color='r')
    axs[1].set_title(f"{title} - Pior Caso")
    axs[1].set_xlabel("Épocas")
    axs[1].grid(True)

    # Ajustar layout
    plt.tight_layout()
    plt.show()