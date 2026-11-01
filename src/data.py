import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple


def load_data(filepath: str) -> np.ndarray:
    """
    Carrega pontos lat/lng de um arquivo CSV.

    O CSV deve ter colunas 'lat' e 'lng'.
    Retorna um array numpy de shape (n, 2).

    Parâmetros:
        filepath: caminho para o arquivo CSV

    Retorna:
        np.ndarray de shape (n, 2)
    """
    # Lê o CSV em um DataFrame do pandas
    df = pd.read_csv(filepath)

    # Seleciona as colunas lat e lng (nessa ordem) e converte para array numpy (n, 2)
    points = df[["lat", "lng"]].to_numpy()

    return points


def prepare_features(points: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    """
    Normaliza os features usando StandardScaler.

    Parâmetros:
        points: array numpy de shape (n, 2) com lat/lng

    Retorna:
        Tupla (features_normalizados, scaler)
        O scaler deve ser guardado para inverter a transformação depois.
    """
    # Cria o normalizador (média 0 e desvio padrão 1 para cada coluna)
    scaler = StandardScaler()

    # Aprende média/desvio dos dados e já aplica a normalização
    features = scaler.fit_transform(points)

    # Devolve também o scaler, usado depois na predição e para desfazer a normalização
    return features, scaler
