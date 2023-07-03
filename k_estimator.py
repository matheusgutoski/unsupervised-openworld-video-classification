import numpy as np
from scipy import sparse
from scipy.spatial.distance import cosine
from scipy.cluster.hierarchy import linkage, fcluster

np.set_printoptions(suppress=True)
from sklearn.metrics import silhouette_score
import hierarchical


def compute_dendrogaps(Z, top_k):
    dists = Z[:, 2]
    difs = np.diff(dists)

    sort_idx = np.argsort(difs)[::-1]
    dendrogap_scores = dists[sort_idx][0:top_k]

    k_dendrogap = []
    silhouettes = []
    for l in dendrogap_scores:
        preds = fcluster(Z, t=l, criterion="distance")
        k_dendrogap.append(np.unique(preds).shape[0])

    return k_dendrogap, difs[sort_idx][0:top_k]


def estimate_dendrogap(x, top_k, normalize_data=True):
    from sklearn.preprocessing import normalize

    if normalize_data:
        x = normalize(x.copy())
    Z = linkage(x, "ward", metric="euclidean")

    return compute_dendrogaps(Z, top_k)


def best_silhouette(x, candidates, metric="cosine"):
    best_sil = -1000
    best_idx = None
    for i, k in enumerate(candidates):
        preds = hierarchical.hierarchical(
            x,
            n_clusters=k,
            affinity="euclidean",
            linkage="ward",
            distance_threshold=None,
            normalize_data=True,
        )
        sil = silhouette_score(x, preds, metric=metric)
        if sil > best_sil:
            best_sil = sil
            best_idx = i

    return candidates[best_idx]
