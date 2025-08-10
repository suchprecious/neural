import numpy as np
import matplotlib.pyplot as plt

def generate_data(n_samples=100):
    X = np.linspace(0, 10, n_samples).reshape(-1, 1)
    y = np.sin(X) + 0.1 * np.random.randn(n_samples, 1)
    return X, y

def rbf(x, c, s):
    return np.exp(-1 / (2 * s**2) * np.linalg.norm(x - c)**2)

def train_rbf(X, y, n_rbf=10, sigma=1.0):
    n_samples, n_features = X.shape

    centers = np.linspace(X.min(), X.max(), n_rbf)

    F = np.zeros((n_samples, n_rbf))
    for i in range(n_samples):
        for j in range(n_rbf):
            F[i, j] = rbf(X[i], centers[j], sigma)

    W = np.dot(np.linalg.pinv(F), y)
    
    return centers, W

def predict_rbf(X, centers, W, sigma=1.0):
    n_samples = X.shape[0]
    n_rbf = len(centers)
    F = np.zeros((n_samples, n_rbf))
    
    for i in range(n_samples):
        for j in range(n_rbf):
            F[i, j] = rbf(X[i], centers[j], sigma)
    
    y_pred = np.dot(F, W)
    return y_pred

X, y = generate_data(n_samples=100)

n_rbf = 10
sigma = 1.0
centers, W = train_rbf(X, y, n_rbf, sigma)

X_test = np.linspace(0, 10, 100).reshape(-1, 1)
y_pred = predict_rbf(X_test, centers, W, sigma)

plt.scatter(X, y, color='blue', label='Обучающие данные')
plt.plot(X_test, y_pred, color='red', label='Аппроксимация RBF')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()