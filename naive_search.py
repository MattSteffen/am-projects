import numpy as np
from sklearn.cluster import KMeans
from typing import List, Tuple
from collections import Counter


########################################################################################
# Brute-Force Vector Search
########################################################################################

def brute_force_search(query_vector, dataset, k):
    """
    Brute force search for the k nearest neighbors of a query vector in a dataset.
    Decent implementation found here:

    dataset: numpy array of shape (n_samples, n_features)
    query_vector: numpy array of shape (n_features,)
    k: number of nearest neighbors to return
    """
    distances = np.linalg.norm(dataset-query_vector, axis=1)
    indices = np.argsort(distances)[:k]
    return indices

def test_brute_force_search():
    dataset = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    query_vector = np.array([1, 1, 1])
    k = 2
    print(brute_force_search(query_vector, dataset, k))


########################################################################################
# FAISS-like Vector Search
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
    

def test_faiss_like_search():
    np.random.seed(42)
    dataset = np.random.rand(1000, 128)
    query_vector = np.random.rand(128)

    faiss = SimpleFAISS(n_subvectors=8, n_clusters=256)
    faiss.train(dataset)

    k = 5
    results = faiss.search(query_vector, k)

    print(f"Query vector: {query_vector}")
    print(f"Top {k} closest vectors:")
    for distance, vector in results:
        print(f"Distance: {distance:.2f}, Vector: {vector[:5]}...")




########################################################################################
# KNN Search
########################################################################################


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a - b)

def knn_search(query_vector: np.ndarray, dataset: List[np.ndarray], k: int) -> List[Tuple[float, np.ndarray]]:
    distances = []
    for vector in dataset:
        distance = euclidean_distance(query_vector, vector)
        distances.append((distance, vector))

    return sorted(distances, key=lambda x: x[0])[:k]

def knn_classify(query_vector: np.ndarray, dataset: List[np.ndarray], labels: List[str], k: int) -> str:
    neighbors = knn_search(query_vector, dataset, k)
    neighbor_labels = [labels[dataset.index(vector)] for _, vector in neighbors]
    return Counter(neighbor_labels).most_common(1)[0][0]

def test_knn_search():
    # Create a sample dataset
    dataset = [
        np.array([1, 2]), np.array([2, 3]), np.array([3, 4]),
        np.array([5, 6]), np.array([6, 7]), np.array([7, 8])
    ]
    labels = ['A', 'A', 'A', 'B', 'B', 'B']
    query_vector = np.array([4, 5])
    k = 3

    # Perform KNN search
    search_results = knn_search(query_vector, dataset, k)
    print(f"Query vector: {query_vector}")
    print(f"Top {k} closest vectors:")
    for distance, vector in search_results:
        print(f"Distance: {distance:.2f}, Vector: {vector}")

    # Perform KNN classification
    classification = knn_classify(query_vector, dataset, labels, k)
    print(f"\nClassification result: {classification}")






########################################################################################
# Kmeans Search
########################################################################################



def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a - b)

def kmeans_clustering(dataset: np.ndarray, k: int, max_iterations: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    # Randomly initialize centroids
    centroids = dataset[np.random.choice(dataset.shape[0], k, replace=False)]

    for _ in range(max_iterations):
        # Assign each data point to the nearest centroid
        distances = np.array([np.linalg.norm(dataset - c, axis=1) for c in centroids])
        cluster_assignments = np.argmin(distances, axis=0)

        # Update centroids
        new_centroids = np.array([dataset[cluster_assignments == i].mean(axis=0) for i in range(k)])

        # Check for convergence
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return centroids, cluster_assignments

def kmeans_search(query_vector: np.ndarray, dataset: np.ndarray, centroids: np.ndarray, cluster_assignments: np.ndarray, k: int) -> List[Tuple[float, np.ndarray]]:
    # Find the nearest centroid to the query vector
    centroid_distances = [euclidean_distance(query_vector, centroid) for centroid in centroids]
    nearest_centroid = np.argmin(centroid_distances)

    # Get points in the nearest cluster
    cluster_points = dataset[cluster_assignments == nearest_centroid]

    # Perform KNN search within the cluster
    distances = [euclidean_distance(query_vector, point) for point in cluster_points]
    sorted_indices = np.argsort(distances)[:k]

    return [(distances[i], cluster_points[i]) for i in sorted_indices]

def test_kmeans_search():
    np.random.seed(42)
    dataset = np.random.rand(1000, 10)  # 1000 vectors of dimension 10
    query_vector = np.random.rand(10)  # Query vector of dimension 10
    k_clusters = 10
    k_neighbors = 5

    # Perform K-means clustering
    centroids, cluster_assignments = kmeans_clustering(dataset, k_clusters)

    # Perform K-means search
    results = kmeans_search(query_vector, dataset, centroids, cluster_assignments, k_neighbors)

    print(f"Query vector: {query_vector}")
    print(f"Top {k_neighbors} nearest neighbors:")
    for distance, vector in results:
        print(f"Distance: {distance:.4f}, Vector: {vector}")

    # Assert that we got the correct number of results
    assert len(results) == k_neighbors, f"Expected {k_neighbors} results, but got {len(results)}"

    # Assert that the distances are in ascending order
    distances = [distance for distance, _ in results]
    assert all(distances[i] <= distances[i+1] for i in range(len(distances)-1)), "Distances are not in ascending order"

    print("All tests passed successfully!")
