import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import geopy.distance


def predict(model: KMeans, scaler: StandardScaler, lat: float, lng: float) -> dict:
    """
    Dado um ponto (lat, lng), retorna o cluster mais próximo e a distância ao centróide.

    Passos esperados:
        1. Normalizar o ponto com o scaler recebido
        2. Predizer o cluster com o modelo
        3. Recuperar o centróide do cluster (invertendo a normalização)
        4. Calcular a distância geodésica em km entre o ponto e o centróide

    Parâmetros:
        model: modelo KMeans treinado
        scaler: StandardScaler usado no treino (para normalizar o ponto de entrada)
        lat: latitude do ponto a ser classificado
        lng: longitude do ponto a ser classificado

    Retorna:
        Dicionário com as chaves:
        - 'cluster': int — índice do cluster mais próximo
        - 'distance_km': float — distância em km ao centróide (arredondado em 4 casas)
    """
    # 1. Normaliza o ponto com o mesmo scaler usado no treino
    point_scaled = scaler.transform([[lat, lng]])

    # 2. Prediz o cluster do ponto
    cluster = int(model.predict(point_scaled)[0])

    # 3. Pega o centróide desse cluster e desfaz a normalização (volta para lat/lng)
    centroid_scaled = model.cluster_centers_[cluster]
    centroid_original = scaler.inverse_transform([centroid_scaled])[0]

    # 4. Distância geodésica (sobre a superfície da Terra) entre o ponto e o centróide, em km
    distance_km = geopy.distance.geodesic(
        (lat, lng),
        (centroid_original[0], centroid_original[1]),
    ).km

    return {
        "cluster": cluster,
        "distance_km": round(float(distance_km), 4),
    }
