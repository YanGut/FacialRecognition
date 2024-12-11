import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import os
from typing import List
from matrices import confusion_matrix, plot_confusion_matrix
from metrics import calculate_metrics, update_results, build_summary, plot_leaning_curves

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

def prepare_data(
    df: pd.DataFrame, 
    input_columns: List[str], 
    target_column: str,
    transpose: bool = False
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

    return X, Y

def plot_training_data(data_frame: pd.DataFrame) -> None:
    """Plota os dados de treinamento."""
    plt.scatter(data_frame[data_frame['spiral'] == 1.0]['x'],
                data_frame[data_frame['spiral'] == 1.0]['y'], 
                color='blue', label='Classe 1.0', alpha=0.7)
    plt.scatter(data_frame[data_frame['spiral'] == -1.0]['x'], 
                data_frame[data_frame['spiral'] == -1.0]['y'], 
                color='red', label='Classe -1.0', alpha=0.7)
    plt.title('Treinamento do Perceptron - Linha de Decisão', fontsize=16)
    plt.xlabel('X', fontsize=14)
    plt.ylabel('Y', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xlim(-20, 20)
    plt.ylim(-20, 20)

def update_decision_boundary(W: np.ndarray, x_axis: np.ndarray) -> None:
    """Atualiza a linha de decisão no gráfico."""
    x2 = -W[1, 0] / W[2, 0] * x_axis + W[0, 0] / W[2, 0]
    x2 = np.nan_to_num(x2)
    plt.plot(x_axis, x2, color='orange', alpha=0.1)
    plt.pause(0.1)

def plot_final_decision_boundary(W: np.ndarray, x_axis: np.ndarray) -> None:
    """Plota a linha de decisão final."""
    x2 = -W[1, 0] / W[2, 0] * x_axis + W[0, 0] / W[2, 0]
    x2 = np.nan_to_num(x2)
    plt.plot(x_axis, x2, color='green', linewidth=2)

def sign(u: float) -> int:
    """
    Função de ativação degrau.

    Args:
        u (float): Soma ponderada.

    Returns:
        int: Saída da função de ativação.
    """
    return 1 if u >= 0 else -1

def simple_perceptron(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    epochs: int = 1000,
    learning_rate: float = 0.01,
    tolerance: float = 1e-4,
    patience: int = 10,
    w_random: bool = True,
    data_frame: pd.DataFrame = None
) -> tuple[np.ndarray, list[float]]:
    """
    Implementa o Perceptron Simples com critérios avançados de parada.

    Args:
        X_train (np.ndarray): Entradas de treinamento, com dimensão (p+1, N), incluindo bias.
        Y_train (np.ndarray): Rótulos de treinamento, com dimensão (1, N).
        epochs (int): Número máximo de épocas.
        learning_rate (float): Taxa de aprendizado.
        tolerance (float): Norma mínima da atualização dos pesos para critério de convergência.
        patience (int): Número de épocas sem melhora antes de ativar o early stopping.
        w_random (bool): Inicializar pesos aleatoriamente (True) ou como zeros (False).
        data_frame (pd.DataFrame, opcional): DataFrame para visualização dos dados.

    Returns:
        tuple[np.ndarray, list[float]]: 
            - Vetor de pesos aprendido, com dimensão (p+1, 1).
            - Histórico de erro quadrático médio por época.
    """
    p, N = X_train.shape
    W = (np.random.random_sample((p, 1)) - 0.5) if w_random else np.zeros((p, 1))
    mse_history = []
    no_improve_count = 0

    # Configuração inicial do gráfico
    if data_frame is not None:
        plt.figure(figsize=(8, 6))
        plot_training_data(data_frame)
        x_axis = np.linspace(-15, 15, 100)

    for epoch in range(epochs):
        errors = []
        weight_update_norm = 0

        for t in range(N):
            x_t = X_train[:, t].reshape(p, 1)
            d_t = Y_train[0, t]

            # Calcula predição e erro
            u_t = W.T @ x_t
            y_t = sign(u_t[0, 0])
            e_t = d_t - y_t
            
            # Atualiza pesos
            weight_update = learning_rate * e_t * x_t / 2
            W += weight_update
            weight_update_norm += np.linalg.norm(weight_update)

            errors.append(e_t ** 2)

        # Calcula erro quadrático médio
        mse = np.mean(errors)
        mse_history.append(mse)

        # Critério de parada: norma da atualização dos pesos
        if weight_update_norm < tolerance:
            print(f"Convergência alcançada na época {epoch + 1} devido à tolerância.")
            break

        # Early stopping baseado no erro
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

def simple_perceptron_test(
    X_test: np.ndarray,
    W: np.ndarray
) -> np.ndarray:
    """
    Realiza a classificação de amostras desconhecidas usando um perceptron simples.

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
    R = 1
    epochs = 1000
    learning_rate = 0.01
    separe_data_with_sklearn = False
    
    columns = ["x", "y", "spiral"]
    df: pd.DataFrame = load_csv_data(filepath="resources/spiral.csv", columns=columns, transpose=False)
    print(df.head())
    
    plot_data(df=df)
    
    data, labels = prepare_data(df=df,
                                input_columns=["x", "y"],
                                target_column="spiral",
                                transpose=True)

    print(data.shape)
    print(labels.shape)
    
    # Monte Carlo
    n_samples = data.shape[1]  # Adjusted to match the shape (3, 1999)
    metrics = {"accuracy": [], "sensitivity": [], "specificity": []}
    results = []

    for i in range(R):
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        data, labels = data[:, indices], labels[indices]
        
        if separe_data_with_sklearn:
            X_train, X_test, Y_train, Y_test = train_test_split(data.T, labels, test_size=0.2, random_state=42)
            X_train, X_test = X_train.T, X_test.T  # Transpose back to (features, samples)
        else:
            N_train = int(0.8 * n_samples)
            X_train, Y_train = data[:, :N_train], labels[:N_train]
            X_test, Y_test = data[:, N_train:], labels[N_train:]

        weights, mse_history = simple_perceptron(
            X_train=X_train,
            Y_train=Y_train.reshape(1, -1),
            epochs=epochs,
            learning_rate=learning_rate,
            w_random=True,
            data_frame=df
        )
        predictions = simple_perceptron_test(X_test, weights)
        update_results(metrics=metrics, results=results, predictions=predictions, Y_test=Y_test, mse_history=mse_history)

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