import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import os
from typing import List, Tuple

def load_csv_data(filepath: str, columns: List[str], transpose: bool, sep: str = ',') -> pd.DataFrame:
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
    if transpose:
        df: pd.DataFrame = pd.read_csv(filepath, header=None, sep=sep).T
    else:
        df: pd.DataFrame = pd.read_csv(filepath, sep=sep)
    
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

def plot_mse_history(history: List[float]) -> None:
    """Plota o histórico do erro quadrático médio."""
    plt.figure(figsize=(8, 6))
    plt.plot(history, marker='o', color='b', label='Erro Quadrático Médio')
    plt.xlabel('Épocas')
    plt.ylabel('Erro Quadrático Médio')
    plt.title('Erro Quadrático Médio por Época', fontsize=16)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

def prepare_data(
    df: pd.DataFrame, 
    input_columns: List[str], 
    target_column: str,
    transpose: bool = False,
    normalize: bool = False,
    test_size: float = 0.2, 
    random_state: int = 42,
) -> dict:
    """
    Prepara o conjunto de dados para redes neurais, organizando entradas, saídas e divisões.

    Args:
        df (pd.DataFrame): DataFrame contendo o conjunto de dados.
        input_columns (list[str]): Lista com os nomes das colunas de entrada.
        target_column (str): Nome da coluna de saída (rótulos).
        transpose (bool): Transpor o DataFrame. Default é False.
        test_size (float): Proporção do conjunto de teste (0 a 1). Default é 0.2 (20%).
        random_state (int): Semente para reprodutibilidade da divisão. Default é 42.

    Returns:
        dict: Um dicionário contendo os conjuntos organizados:
            - 'X_train': Entradas para treinamento.
            - 'X_test': Entradas para teste.
            - 'Y_train': Rótulos para treinamento.
            - 'Y_test': Rótulos para teste.
    """
    X = df[input_columns].values
    Y = df[target_column].values
    
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

    return {
        'X': X,
        'Y': Y,
    }

def evaluate_accuracy(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    """
    Avalia a precisão das predições.

    Args:
        Y_true (np.ndarray): Valores reais das classes.
        Y_pred (np.ndarray): Valores previstos.

    Returns:
        float: Precisão em porcentagem.
    """
    correct = np.sum(Y_true == Y_pred)
    total = Y_true.size
    return (correct / total) * 100

def sign(u: float) -> int:
    """
    Função de ativação degrau.

    Args:
        u (float): Soma ponderada.

    Returns:
        int: Saída da função de ativação.
    """
    return 1 if u >= 0 else -1

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

def adaline_train(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    epochs: int = 1000,
    learning_rate: float = 0.1,
    w_random: bool = True,
    data_frame: pd.DataFrame = None,
    precision: float = 0.2
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
    history_of_mse = []
    previous_mse, current_mse = 0, 1

    # Inicialização dos pesos
    W = (np.random.random_sample((p, 1)) - 0.5) if w_random else np.zeros((p, 1))

    # Configuração inicial do gráfico
    if data_frame is not None:
        plt.figure(figsize=(8, 6))
        plot_training_data(data_frame)
        x_axis = np.linspace(-15, 15, 100)

    # Treinamento
    for epoch in range(epochs):
        if np.abs(previous_mse - current_mse) < precision:
            print(f"Treinamento convergiu em {epoch} épocas.")
            break

        previous_mse = current_mse
        current_mse = compute_root_mean_square_error(X_train, Y_train, W)
        history_of_mse.append(current_mse)

        for t in range(N):
            x_t = X_train[:, t].reshape(p, 1)
            d_t = Y_train[0, t]

            # Atualiza os pesos com a regra do Adaline
            u_t = W.T @ x_t
            error_t = d_t - u_t
            W += learning_rate * error_t * x_t

        # Atualização da linha de decisão
        if data_frame is not None:
            update_decision_boundary(W, x_axis)

    # Gráficos finais
    if data_frame is not None:
        plot_final_decision_boundary(W, x_axis)
    plot_mse_history(history_of_mse)

    return W, history_of_mse

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
    predictions = np.zeros((1, M))

    for t in range(M):
        x_unknown = X_test[:, t].reshape(p, 1)
        
        # Operação de decisão
        u = W.T @ x_unknown
        y = sign(u[0, 0])

        # Classificação
        predictions[0, t] = y

        if y == -1:
            print(f"Amostra {t} pertence à classe A.")
        else:
            print(f"Amostra {t} pertence à classe B.")

    return predictions


def main() -> None:
    columns = ["x", "y", "spiral"]
    df: pd.DataFrame = load_csv_data(filepath = "resources/spiral.csv", columns = columns, transpose = False)
    print(df.head())
    
    plot_data(df=df)
    
    data = prepare_data(df = df,
                        input_columns=["x", "y"],
                        target_column="spiral",
                        transpose=True,
                        normalize=True,
                        test_size=0.2,
                        random_state=42)

    print(data['X'].shape)
    print(data['Y'].shape)
    
    weights, history_of_error = adaline_train(X_train = data['X'],
                                Y_train = data['Y'].reshape(1, -1),
                                epochs = 1000,
                                learning_rate = 1e-4,
                                w_random = True,
                                data_frame = df,
                                precision=1e-8)
    
    print("Pesos aprendidos:\n", weights)
    
    predictions = adaline_test(X_test = data['X'], W = weights)
    
    accuracy = evaluate_accuracy(Y_true = data['Y'], Y_pred = predictions)
    print(f"Precisao no teste: {accuracy:.2f}%")

if __name__ == '__main__':
    main()