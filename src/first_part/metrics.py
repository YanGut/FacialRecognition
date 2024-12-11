import numpy as np
from typing import List, Tuple, Dict, Union
import pandas as pd
import matplotlib.pyplot as plt
from matrices import confusion_matrix

def calculate_metrics(predictions: np.ndarray, Y_test: np.ndarray) -> (float, float, float):
    """
    Calcula metricas de desempenho para o modelo.
    
    Args:
        predictions (np.ndarray): Vetor de predições.
        Y_test (np.ndarray): Vetor de rótulos reais.
    
    Returns:
        Tuple[float, float, float]: Acurácia, Sensibilidade e Especificidade.
    """
    
    acuracy = np.mean(predictions == Y_test)
    true_positive = np.sum((predictions == 1) & (Y_test == 1))
    true_negative = np.sum((predictions == -1) & (Y_test == -1))
    false_positive = np.sum((predictions == 1) & (Y_test == -1))
    false_negative = np.sum((predictions == -1) & (Y_test == 1))
    
    sensitivity = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0
    
    return acuracy, sensitivity, specificity

def update_results(
    metrics: Tuple[float, float, float],
    results: List[Dict[str, Union[float, np.ndarray]]],
    predictions: np.ndarray,
    Y_test: np.ndarray,
    mse_history: List[float]
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
    
    accuracy, sensitivity, specificity = calculate_metrics(predictions, Y_test)

    metrics["accuracy"].append(accuracy)
    metrics["sensitivity"].append(sensitivity)
    metrics["specificity"].append(specificity)

    results.append({"accuracy": accuracy, "conf_matrix": confusion_matrix(Y_test, predictions), "mse": mse_history})

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