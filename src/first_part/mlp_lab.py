import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import os
from typing import List, Tuple, Dict, Union
from matrices import confusion_matrix, plot_confusion_matrix
from metrics import calculate_metrics, update_results, build_summary, plot_leaning_curves

def load_csv_data(filepath: str, columns: List[str], sep: str = ',') -> pd.DataFrame:
    """
    Carrega o dataset EMG e organiza as colunas.

    Args:
        filepath (str): Caminho para o arquivo.
        columns (List[str]): Lista com os nomes das colunas.
        transpose (bool): Transpor o DataFrame.
        sep (str): Separador dos dados.

    Returns:
        pd.DataFrame: DataFrame com dados dos sensores e classes.
    """
    
    df: pd.DataFrame = pd.read_csv(filepath, header=None, sep=sep)
    
    df.columns = columns
    
    return df

def plot_data(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 6))

    plt.scatter(df[df['spiral'] == 1.0]['x'], 
                df[df['spiral'] == 1.0]['y'], 
                color='blue', label='Classe 1.0', alpha=0.7)

    plt.scatter(df[df['spiral'] == -1.0]['x'], 
                df[df['spiral'] == -1.0]['y'], 
                color='red', label='Classe -1.0', alpha=0.7)

    plt.title('Gráfico de Dispersão (Espalhamento)', fontsize=16)
    plt.xlabel('X', fontsize=14)
    plt.ylabel('Y', fontsize=14)

    plt.legend()

    plt.grid(alpha=0.3)
    plt.show()

def plot_training_data(data_frame: pd.DataFrame) -> None:
    """Plota os dados de treinamento."""
    plt.scatter(data_frame[data_frame['spiral'] == 1.0]['x'],
                data_frame[data_frame['spiral'] == 1.0]['y'], 
                color='blue', label='Classe 1.0', alpha=0.7)
    plt.scatter(data_frame[data_frame['spiral'] == -1.0]['x'], 
                data_frame[data_frame['spiral'] == -1.0]['y'], 
                color='red', label='Classe -1.0', alpha=0.7)
    plt.title('Treinamento do Adaline - Linha de Decisão', fontsize=16)
    plt.xlabel('X', fontsize=14)
    plt.ylabel('Y', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xlim(-20, 20)
    plt.ylim(-20, 20)


def prepare_data(
    df: pd.DataFrame, 
    input_columns: List[str], 
    target_column: str,
    transpose: bool = False,
    normalize: bool = False,
) -> dict:
    """
    Prepara o conjunto de dados para redes neurais, organizando entradas, saídas e divisões.

    Args:
        df (pd.DataFrame): DataFrame contendo o conjunto de dados.
        input_columns (list[str]): Lista com os nomes das colunas de entrada.
        target_column (str): Nome da coluna de saída (rótulos).
        transpose (bool): Transpor o DataFrame. Default é False.

    Returns:
        dict: Um dicionário contendo os conjuntos organizados:
            - 'X_train': Entradas para treinamento.
            - 'X_test': Entradas para teste.
            - 'Y_train': Rótulos para treinamento.
            - 'Y_test': Rótulos para teste.
    """    
    X = df[[
            input_columns[0],
            input_columns[1]
        ]].to_numpy()
    Y = df[target_column[0]].to_numpy().reshape(-1, 1)
    
    if transpose:
        X = X.T
        Y = Y.T
        
        p, N = X.shape
        X = np.concatenate((
            -np.ones((1, N)),
            X
        ))
        
    else:
        N, p = X.shape
        X = np.concatenate((
            -np.ones((N, 1)),
            X
        ), axis = 1)
    
    if normalize:
        X = 2 * (X - X.min()) / (X.max() - X.min()) - 1

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

def mlp_train(
    data: np.ndarray,
    labels: np.ndarray,
    hidden_units: int,
    last_layer_units: int = 1,
    learning_rate: float = 0.01,
    epochs: int = 100,
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
            print(f"Converged at epoch {epoch}.")
            break
        
        # Backpropagation
        deltas = [None] * len(weights)
        deltas[-1] = (y_pred - labels) * activation_derivate(u=y_pred, logistic=False, hyperbolic=True)
        
        for l in range(len(weights) - 2, -1, -1):
            a_with_bias = np.vstack([np.ones((1, activations[l + 1].shape[1])), activations[l + 1]])
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
    print(f"Final output: {final_output.shape}")
    predictions = np.where(final_output >= 0, 1, -1)
    
    return predictions.flatten()

def main() -> None:
    R = 1
    epochs = 1000
    learning_rate = 0.01
    tolerance = 1e-3
    patience = 10
    hidden_units = [8, 8, 8, 8, 8, 8, 8, 8]
    last_layer_units = 1
    separe_data_with_sklearn = False
    
    columns = ["x", "y", "spiral"]
    df: pd.DataFrame = load_csv_data(filepath = "resources/spiral.csv", columns = columns)
    print(df.head())
    
    data, labels = prepare_data(
        df=df, 
        input_columns=["x", "y"], 
        target_column=["spiral"],
        transpose=False,
        normalize=True
    )
    
    plot_data(df)

    # Monte Carlo
    n_samples = data.shape[0]
    metrics = {"accuracy": [], "sensitivity": [], "specificity": []}
    results = []

    for i in range(R):
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
        update_results(metrics=metrics, results=results, predictions=predictions, Y_test=Y_test, mse_history=mse_history)

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
