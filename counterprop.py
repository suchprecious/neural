import numpy as np
import matplotlib.pyplot as plt

def generate_data(n_samples=100):
    X = np.linspace(-10, 10, n_samples).reshape(-1, 1)
    y = np.sin(X) + 0.1 * np.random.randn(n_samples, 1)
    return X, y

def initialize_kohonen_weights(input_dim, n_neurons):
    return np.random.rand(n_neurons, input_dim)

def initialize_grossberg_weights(n_neurons, output_dim):
    return np.random.rand(n_neurons, output_dim)

def find_winner(weights, input_vector):
    distances = np.linalg.norm(weights - input_vector, axis=1)
    return np.argmin(distances)

def train_kohonen(X, n_neurons, learning_rate=0.1, n_epochs=100):
    input_dim = X.shape[1]
    weights = initialize_kohonen_weights(input_dim, n_neurons)
    
    for epoch in range(n_epochs):
        for input_vector in X:
            winner = find_winner(weights, input_vector)
            weights[winner] += learning_rate * (input_vector - weights[winner])
    
    return weights

def train_grossberg(X, y, kohonen_weights, learning_rate=0.1, n_epochs=100):
    n_neurons = kohonen_weights.shape[0]
    output_dim = y.shape[1]
    grossberg_weights = initialize_grossberg_weights(n_neurons, output_dim)
    
    for epoch in range(n_epochs):
        for input_vector, target in zip(X, y):
            winner = find_winner(kohonen_weights, input_vector)
            grossberg_weights[winner] += learning_rate * (target - grossberg_weights[winner])
    
    return grossberg_weights

def predict(X, kohonen_weights, grossberg_weights):
    predictions = []
    for input_vector in X:
        winner = find_winner(kohonen_weights, input_vector)
        predictions.append(grossberg_weights[winner])
    return np.array(predictions)

X, y = generate_data(n_samples=100)

n_neurons = 10
kohonen_weights = train_kohonen(X, n_neurons, learning_rate=0.1, n_epochs=100)

grossberg_weights = train_grossberg(X, y, kohonen_weights, learning_rate=0.1, n_epochs=100)

X_test = np.linspace(-10, 10, 100).reshape(-1, 1)
y_pred = predict(X_test, kohonen_weights, grossberg_weights)

plt.scatter(X, y, color='blue', label='Обучающие данные')
plt.plot(X_test, y_pred, color='red', label='Аппроксимация CPN')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()