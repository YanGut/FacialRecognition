import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import os
from typing import List, Tuple
from matrices import calculate_confusion_matrix, plot_confusion_matrix
from metrics import calculate_metrics, update_results, build_summary, plot_leaning_curves
import cv2

def load_images(pasta_raiz: str, dimensao: int, C: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carrega e processa as imagens para reconhecimento facial.

    Args:
        pasta_raiz (str): Caminho para a pasta raiz contendo as subpastas de cada pessoa.
        dimensao (int): Dimensão para redimensionar as imagens (dimensao x dimensao).
        C (int): Número total de classes (pessoas).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Matriz de dados X (p x N) e matriz de rótulos Y (C x N).
    """
    caminho_pessoas = [x[0] for x in os.walk(pasta_raiz)]
    caminho_pessoas.pop(0)

    X = np.empty((dimensao * dimensao, 0))  # Matriz de dados
    Y = np.empty((C, 0))  # Matriz de rótulos

    for i, pessoa in enumerate(caminho_pessoas):
        imagens_pessoa = os.listdir(pessoa)
        for imagem in imagens_pessoa:
            caminho_imagem = os.path.join(pessoa, imagem)
            imagem_original = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)
            imagem_redimensionada = cv2.resize(imagem_original, (dimensao, dimensao))

            # Vetorizando a imagem
            x = imagem_redimensionada.flatten()

            # Empilhando amostra para criar a matriz X
            X = np.concatenate((X, x.reshape(dimensao * dimensao, 1)), axis=1)

            # One-hot encoding
            y = one_hot_encode(i, C)
            Y = np.concatenate((Y, y), axis=1)

    return X, Y

def one_hot_encode(index: int, C: int) -> np.ndarray:
    """
    Realiza o one-hot encoding para um índice de classe.

    Args:
        index (int): Índice da classe.
        C (int): Número total de classes.

    Returns:
        np.ndarray: Vetor one-hot encoded.
    """
    y = -np.ones((C, 1))
    y[index, 0] = 1
    return y

def update_decision_boundary(W: np.ndarray, x_axis: np.ndarray) -> None:
    """Atualiza a linha de decisão no gráfico."""
    if W[2, 0] != 0:
        x2 = -W[1, 0] / W[2, 0] * x_axis + W[0, 0] / W[2, 0]
        x2 = np.nan_to_num(x2)
        plt.plot(x_axis, x2, color='orange', alpha=0.1)
        plt.pause(0.1)

def plot_final_decision_boundary(W: np.ndarray, x_axis: np.ndarray) -> None:
    """Plota a linha de decisão final."""
    if W[2, 0] != 0:
        x2 = -W[1, 0] / W[2, 0] * x_axis + W[0, 0] / W[2, 0]
        x2 = np.nan_to_num(x2)
        plt.plot(x_axis, x2, color='green', linewidth=2)
    plt.show()

def prepare_data(
    X: np.ndarray,
    Y: np.ndarray,
    transpose: bool = False,
    normalize: bool = False
) -> dict:
    """
    Prepara os dados para treinamento e teste.
    
    Args:
        X (np.ndarray): Matriz de dados, com dimensão (p, N).
        Y (np.ndarray): Matriz de rótulos, com dimensão (C, N).
        transpose (bool): Transpor os dados.
        normalize (bool): Normalizar os dados.
    
    Returns:
        dict: Dicionário contendo as matrizes de dados e rótulos preparadas.
    """     
    if normalize:
        min_values = np.min(X, axis=0)
        max_values = np.max(X, axis=0)
        normalized_data = (X - min_values) / (max_values - min_values)
        X = 2 * normalized_data - 1
    
    if transpose:
        X = X.T
        Y = Y.T

    return X, Y

def sign(u: float) -> int:
    """
    Função de ativação degrau.

    Args:
        u (float): Soma ponderada.

    Returns:
        int: Saída da função de ativação.
    """
    return 1 if u >= 0 else -1

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def adaline_train(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    epochs: int = 1000,
    learning_rate: float = 0.1,
    w_random: bool = True,
    data_frame: pd.DataFrame = None,
    precision: float = 0.2,
    tolerance: float = 1e-4,
) -> Tuple[np.ndarray, List[float]]:
    """
    Treina um modelo Adaline usando o conjunto de dados fornecido.

    Args:
        X_train (np.ndarray): Entradas de treinamento, com dimensão (p+1, N), incluindo bias.
        Y_train (np.ndarray): Rótulos de treinamento, com dimensão (1, N).
        epochs (int): Número máximo de épocas.
        learning_rate (float): Taxa de aprendizado.
        w_random (bool): Define se os pesos serão inicializados aleatoriamente (True) ou como zeros (False).
        data_frame (pd.DataFrame, opcional): DataFrame para visualização dos dados (necessita das colunas 'x', 'y' e 'spiral').
        precision (float): Precisão mínima para a convergência baseada na diferença do erro quadrático médio.

    Returns:
        Tuple[np.ndarray, List[float]]: 
            - Vetor de pesos aprendido, com dimensão (p+1, 1).
            - Histórico do erro quadrático médio por época.
    """
    # Inicialização
    p, N = X_train.shape
    C = Y_train.shape[1]
    W = (np.random.random_sample((N + 1, C)) - 0.5) if w_random else np.zeros((N + 1, C))
    previous_mse, current_mse = 0, 1
    no_improvement = 0
    mse_history = []
    
    # Adiciona termo de bias
    X_train = np.hstack((np.ones((p, 1)), X_train))

    # Configuração inicial do gráfico
    if data_frame is not None:
        plt.figure(figsize=(8, 6))
        plot_training_data(data_frame)
        x_axis = np.linspace(-15, 15, 100)

    # Treinamento
    print(f"X_train shape: {X_train.shape}")
    print(f"W shape: {W.shape}")
    for epoch in range(epochs):
        # Linear output for all classes
        linear_output = np.dot(X_train, W)
        predictions = softmax(linear_output)  # Multi-class prediction using softmax

        # Calculate errors and MSE
        errors = Y_train - predictions
        mse = np.mean(errors ** 2)
        mse_history.append(mse)

        # Update weights
        weight_update = learning_rate * np.dot(X_train.T, errors) / p
        W += weight_update

        # Stopping criteria based on weight update norm
        if np.linalg.norm(weight_update) < tolerance:
            print(f"Convergência alcançada na época {epoch + 1} devido à tolerância.")
            break

        # Early stopping based on error improvement
        if mse == 0:
            no_improve_count += 1
            if no_improve_count >= patience:
                print("Early stopping ativado devido a nenhuma melhora no erro.")
                break
        else:
            no_improve_count = 0

        # Atualiza a linha de decisão no gráfico
        if data_frame is not None and W[2, 0] != 0:
            update_decision_boundary(W, x_axis)

    # Linha final após o término do treinamento
    if data_frame is not None and W[2, 0] != 0:
        plot_final_decision_boundary(W, x_axis)

    if data_frame is not None:
        plt.show()

    return W, mse_history

def adaline_test(
    X_test: np.ndarray,
    W: np.ndarray
) -> np.ndarray:
    """
    Realiza a classificação de amostras desconhecidas usando o ANALINE.

    Args:
        X_test (np.ndarray): Conjunto de entradas para teste, com dimensão (p+1, M), incluindo bias.
        W (np.ndarray): Vetor de pesos aprendido na fase de treinamento, com dimensão (p+1, 1).

    Returns:
        np.ndarray: Vetor de predições, com dimensão (1, M), onde cada valor é -1 ou 1, 
                    indicando a classe prevista para cada amostra.
    """
    p, M = X_test.shape
    X_train = np.hstack((np.ones((p, 1)), X_test))
    linear_output = np.dot(X_train, W)
    predictions = softmax(linear_output)
    
    return np.argmax(predictions, axis=1)


def main() -> None:
    R = 5
    epochs = 1000
    learning_rate = 0.01
    precision = 1e-6
    tolerance = 1e-4
    separe_data_with_sklearn = False
    
    pasta_raiz = "resources/RecFac"
    dimensao = 50
    C = 20

    X, Y = load_images(pasta_raiz, dimensao, C)    
    data, labels = prepare_data(X, Y, transpose=True, normalize=True)
    
    # Monte Carlo
    n_samples = data.shape[0]
    metrics = {"accuracy": [], "sensitivity": [], "specificity": []}
    results = []

    for i in range(R):
        print(f"Starting iteration {i + 1}...")
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        data, labels = data[indices], labels[indices]  # Corrected indexing
        
        if separe_data_with_sklearn:
            X_train, X_test, Y_train, Y_test = train_test_split(data.T, labels.T, test_size=0.2, random_state=42)
            X_train, X_test = X_train.T, X_test.T  # Transpose back to (features, samples)
            Y_train, Y_test = Y_train.T, Y_test.T  # Transpose back to (samples, classes)
        else:
            N_train = int(0.8 * n_samples)
            X_train, Y_train = data[:N_train], labels[:N_train]
            X_test, Y_test = data[N_train:], labels[N_train:]
    
        weights, mse_history = adaline_train(
            X_train = X_train,
            Y_train = Y_train,
            epochs = epochs,
            learning_rate = learning_rate,
            w_random = True,
            precision = precision,
            tolerance = tolerance
        )
        predictions = adaline_test(X_test = X_test, W = weights)
        update_results(metrics=metrics, results=results, predictions=predictions, Y_test=Y_test, mse_history=mse_history, n_classes=C)
        
        if (i + 1) % 10 == 0:
            print(f"Finished iteration {i + 1}.")
    
    bestPerceptron = results[np.argmax([r["accuracy"] for r in results])]
    worstPerceptron = results[np.argmin([r["accuracy"] for r in results])]
    summary = build_summary(metrics)

    print(f"\n Resume of MLP metrics: \n")
    print(summary)
    plot_confusion_matrix(
        bestPerceptron["conf_matrix"],
        worstPerceptron["conf_matrix"],
        "Perceptron"
    )
    plot_leaning_curves(
        bestPerceptron["mse"],
        worstPerceptron["mse"],
        "Perceptron"
    )

if __name__ == '__main__':
    main()