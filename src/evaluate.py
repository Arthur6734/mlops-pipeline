import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def evaluate_model(model: KMeans, features: np.ndarray) -> dict:
    """
    Avalia o modelo treinado e retorna um dicionário com as métricas.

    Parâmetros:
        model: modelo KMeans treinado
        features: array numpy normalizado usado no treino

    Retorna:
        Dicionário com as chaves:
        - 'inertia': float — soma das distâncias quadráticas ao centróide mais próximo
        - 'silhouette_score': float — qualidade dos clusters (entre -1 e 1, maior é melhor)
        - 'n_clusters': int — número de clusters do modelo
    """
    # Inertia: soma das distâncias quadráticas de cada ponto ao seu centróide (menor = clusters mais compactos)
    inertia = model.inertia_

    # Silhouette: compara a distância ao próprio cluster com a distância ao cluster vizinho
    sil = silhouette_score(features, model.labels_)

    # Monta o dicionário de métricas (float/int nativos para salvar em JSON sem problemas)
    metrics = {
        "inertia": round(float(inertia), 4),
        "silhouette_score": round(float(sil), 4),
        "n_clusters": int(model.n_clusters),
    }

    return metrics
