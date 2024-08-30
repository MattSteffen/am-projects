import numpy as np
from sklearn.cluster import KMeans
from typing import List, Tuple


########################################################################################
# Faiss
########################################################################################

class SimpleFAISS:
    def __init__(self, n_subvectors: int, n_clusters: int):
        self.n_subvectors = n_subvectors
        self.n_clusters = n_clusters
        self.centroids = None
        self.encoded_dataset = None
        self.dataset = None

    def train(self, dataset: np.ndarray):
        self.dataset = dataset
        vector_dim = dataset.shape[1]
        subvector_dim = vector_dim // self.n_subvectors

        self.centroids = []
        for i in range(self.n_subvectors):
            start = i * subvector_dim
            end = (i + 1) * subvector_dim
            subvectors = dataset[:, start:end]

            kmeans = KMeans(n_clusters=self.n_clusters, n_init=10)
            kmeans.fit(subvectors)
            self.centroids.append(kmeans.cluster_centers_)

        self.encoded_dataset = self._encode_vectors(dataset)

    def _encode_vectors(self, vectors: np.ndarray) -> np.ndarray:
        encoded = np.zeros((vectors.shape[0], self.n_subvectors), dtype=int)
        subvector_dim = vectors.shape[1] // self.n_subvectors

        for i in range(self.n_subvectors):
            start = i * subvector_dim
            end = (i + 1) * subvector_dim
            subvectors = vectors[:, start:end]

            distances = np.linalg.norm(subvectors[:, np.newaxis, :] - self.centroids[i], axis=2)
            encoded[:, i] = np.argmin(distances, axis=1)

        return encoded

    def search(self, query_vector: np.ndarray, k: int) -> List[Tuple[float, np.ndarray]]:
        encoded_query = self._encode_vectors(query_vector.reshape(1, -1))[0]
        distances = np.zeros(len(self.encoded_dataset))

        for i in range(self.n_subvectors):
            distances += (self.encoded_dataset[:, i] != encoded_query[i]).astype(int)

        top_k_indices = np.argsort(distances)[:k]
        results = [(distances[i], self.dataset[i]) for i in top_k_indices]
        return results

