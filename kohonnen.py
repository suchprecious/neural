import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

def generate_data(n_samples=1000, n_features=2, n_clusters=4):
    X, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, cluster_std=1.0, random_state=42)
    return X

def initialize_weights(input_dim, map_size):
    return np.random.rand(map_size[0], map_size[1], input_dim)

def euclidean_distance(x1, x2):
    return np.linalg.norm(np.array(x1) - np.array(x2))

def find_winner_neuron(weights, input_vector):
    distances = np.linalg.norm(weights - input_vector, axis=2)
    return np.unravel_index(np.argmin(distances), distances.shape)

def update_weights(weights, input_vector, winner, learning_rate, sigma):
    map_size = weights.shape[:2]
    for i in range(map_size[0]):
        for j in range(map_size[1]):
            distance = euclidean_distance(np.array([i, j]), np.array(winner))
            gaussian = np.exp(-distance**2 / (2 * sigma**2))
            weights[i, j] += learning_rate * gaussian * (input_vector - weights[i, j])

def train_koh(X, map_size, n_epochs=100, learning_rate_init=0.1, sigma_init=1.0):
    input_dim = X.shape[1]
    weights = initialize_weights(input_dim, map_size)
    
    for epoch in range(n_epochs):
        learning_rate = learning_rate_init * np.exp(-epoch / n_epochs)
        sigma = sigma_init * np.exp(-epoch / n_epochs)
        
        for input_vector in X:
            winner = find_winner_neuron(weights, input_vector)
            update_weights(weights, input_vector, winner, learning_rate, sigma)
    
    return weights

def visualize_koh(weights, X):
    plt.figure(figsize=(10, 10))
    plt.scatter(X[:, 0], X[:, 1], c='blue', alpha=0.5)
    
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            plt.plot(weights[i, j, 0], weights[i, j, 1], 'ro')
            if i < weights.shape[0] - 1:
                plt.plot([weights[i, j, 0], weights[i + 1, j, 0]], [weights[i, j, 1], weights[i + 1, j, 1]], 'r-')
            if j < weights.shape[1] - 1:
                plt.plot([weights[i, j, 0], weights[i, j + 1, 0]], [weights[i, j, 1], weights[i, j + 1, 1]], 'r-')
    
    plt.title('Карты Кохоннена')
    plt.show()

X = generate_data(n_samples=1000, n_features=2, n_clusters=4)

map_size = (10, 10)
weights = train_koh(X, map_size, n_epochs=100, learning_rate_init=0.1, sigma_init=1.0)

visualize_koh(weights, X)