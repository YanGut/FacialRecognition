import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import os
from typing import List, Tuple, Dict, Union
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

def compute_root_mean_square_error(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    w: np.ndarray
) -> float:
    """
    Calcula o erro quadrático médio.

    Args:
        X_train (np.ndarray): Entradas para treinamento.
        Y_train (np.ndarray): Rótulos para treinamento.
        w (np.ndarray): Vetor de pesos.

    Returns:
        float: Erro quadrático médio.
    """
    p_1, N = X_train.shape
    square_error = 0

    for t in range(N):
        x_t = X_train[:, t].reshape(p_1, 1)
        u_t = (w.T @ x_t)[0, 0]
        d_t = Y_train[0, t]
        square_error += (d_t - u_t)**2

    return square_error / (2 * N)

def activation_function(
    u: np.ndarray, 
    logistic: bool = True, 
    hyperbolic: bool = False
) -> np.ndarray:
    """
    Função de ativação para a rede MLP. Pode ser logística ou hiperbólica.
    
    Args:
        u (np.ndarray): Vetor de entradas.
        logistic (bool): Função logística. Default é True.
        hyperbolic (bool): Função hiperbólica. Default é False.
    
    Returns:
        np.ndarray: Vetor de saídas.
    """    
    if logistic:
        return (u - np.min(u)) / (np.max(u) - np.min(u))
    
    if hyperbolic:
        return np.tanh(u)
        # return 2 * ((u - np.min(u)) / (np.max(u) - np.min(u))) - 1
    
    raise ValueError("Either 'logistic' or 'hyperbolic' must be True.")

def activation_derivate(
    u: np.ndarray, 
    logistic: bool = True, 
    hyperbolic: bool = False
) -> np.ndarray:
    """
    Derivada da função de ativação para a rede MLP. Pode ser logística ou hiperbólica.
    
    Args:
        u (np.ndarray): Vetor de entradas.
        logistic (bool): Função logística. Default é True.
        hyperbolic (bool): Função hiperbólica. Default é False.
    
    Returns:
        np.ndarray: Vetor de saídas.
    """    
    if logistic:
        return u * (1 - u)
    
    if hyperbolic:
        return 1 - u**2
    
    raise ValueError("Either 'logistic' or 'hyperbolic' must be True.")

activation_derivative = lambda a: 1 - a**2

def mlp_train(
    data: np.ndarray,
    labels: np.ndarray,
    hidden_units: int,
    last_layer_units: int = 20,
    learning_rate: float = 0.0001,
    epochs: int = 1000,
    tolerance: float = 1e-3,
    patience: int = 10,
    transpose: bool = False
):
    """
    Treina uma rede MLP.
    
    Args:
        data (np.ndarray): Dados de entrada.
        labels (np.ndarray): Rótulos.
        hidden_units (int): Número de neurônios na camada oculta.
        last_layer_units (int): Número de neurônios na última camada. Default é 1.
        learning_rate (float): Taxa de aprendizado. Default é 0.01.
        epochs (int): Número máximo de épocas. Default é 100.
        tolerance (float): Tolerância para convergência. Default é 1e-3.
        patience (int): Paciência para early stopping. Default é 10.
        transpose (bool): Transpor o conjunto de dados. Default é False.
    
    Returns:
        tuple: Tupla contendo os pesos treinados e o histórico de erro.
    """
    
    if transpose:
        data = data.T # Tranpose to format (features, samples)
        labels = labels.T # Tranpose to format (features, samples)
    
    n_features, n_samples = data.shape
    
    layers = [n_features] + hidden_units + [last_layer_units]
    
    weights = [
        np.random.randn(layers[i + 1], layers[i] + 1) * np.sqrt(2 / layers[i]) 
        for i in range(len(layers) - 1)
    ]
        
    mse_history = []
    no_improvement = 0
    
    for epoch in range(epochs):
        # Forward pass
        activations = [data]
        z_s = []

        for w in range(len(weights)):
            a_with_bias = np.vstack([np.ones((1, activations[-1].shape[1])), activations[-1]])
            z = np.dot(weights[w], a_with_bias)
            z_s.append(z)
            a = activation_function(u=z, logistic=False, hyperbolic=True)
            activations.append(a)

        y_pred = activations[-1]
        mse = np.mean(np.sum((y_pred - labels) ** 2, axis=1))
        mse_history.append(mse)
        
        if epoch > 0 and mse >= mse_history[-2]:
            no_improvement += 1
            if no_improvement >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break
        else:
            no_improvement = 0
        
        if epoch > 0 and abs(mse_history[-1] - mse_history[-2]) < tolerance:
            print(f"Converged at epoch {epoch + 1}.")
            break
        
        # Backpropagation
        deltas = [None] * len(weights)
        deltas[-1] = (y_pred - labels) * activation_derivate(u=y_pred, logistic=False, hyperbolic=True)
        
        for l in range(len(weights) - 2, -1, -1):
            a_with_bias = np.vstack([np.ones((1, activations[l + 1].shape[1])), activations[l + 1]])
            
            # print(f"Activation derivative: {activation_derivate(u=activations[l + 1], logistic=False, hyperbolic=True).shape}")
            print(f"w: {weights[l + 1][:, 1:].shape}")
            deltas[l] = np.dot(weights[l + 1][:, 1:].T, deltas[l + 1]) * activation_derivate(u=activations[l + 1], logistic=False, hyperbolic=True)
        
        for l in range(len(weights)):
            a_with_bias = np.vstack([np.ones((1, activations[l].shape[1])), activations[l]])
            weights[l] -= learning_rate * np.dot(deltas[l], a_with_bias.T) / n_samples
    
    return weights, mse_history

def mlp_predict(
    data: np.ndarray,
    weights: List[np.ndarray]
) -> np.ndarray:
    """
    Realiza predições com a rede MLP.
    
    Args:
        data (np.ndarray): Dados de entrada.
        weights (List[np.ndarray]): Lista de pesos.
    
    Returns:
        np.ndarray: Vetor de predições.
    """
    
    data = data.T # Tranpose to format (features, samples)
    activations = data
    
    for w in range(len(weights)):
        a_with_bias = np.vstack([np.ones((1, activations.shape[1])), activations])
        z = np.dot(w, a_with_bias)
        activations = activation_function(u=z, logistic=False, hyperbolic=True)
    
    final_output = activations.T
    predictions = np.argmax(final_output, axis=1)
    
    return predictions

def main() -> None:
    R = 5
    epochs = 1000
    learning_rate = 0.01
    tolerance = 1e-4
    patience = 10
    hidden_units = [50, 50, 50]
    last_layer_units = 20
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
        data, labels = data[indices], labels[indices]
        
        if separe_data_with_sklearn:
            X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
        else:
            N_train = int(0.8 * n_samples)
            X_train, Y_train = data[:N_train], labels[:N_train]
            X_test, Y_test = data[N_train:], labels[N_train:]
            
        weights, mse_history = mlp_train(
            data=X_train,
            labels=Y_train,
            hidden_units=hidden_units,
            last_layer_units=last_layer_units,
            learning_rate=learning_rate,
            epochs=epochs,
            tolerance=tolerance,
            patience=patience,
            transpose=True
        )
        predictions = mlp_predict(data=X_test, weights=weights)
        update_results(metrics=metrics, results=results, predictions=predictions, Y_test=Y_test, mse_history=mse_history, n_classes=C)

        if (i + 1) % 10 == 0:
            print(f"Finished iteration {i + 1}.")

    bestMlp = results[np.argmax([r["accuracy"] for r in results])]
    worstMlp = results[np.argmin([r["accuracy"] for r in results])]
    summary = build_summary(metrics)

    print(f"\n Resume of MLP metrics: \n")
    print(summary)
    plot_confusion_matrix(
        bestMlp["conf_matrix"],
        worstMlp["conf_matrix"],
        "MLP"
    )
    plot_leaning_curves(
        bestMlp["mse"],
        worstMlp["mse"],
        "MLP"
    )

if __name__ == "__main__":
    main()
