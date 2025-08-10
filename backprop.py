import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

digits = load_digits()
X, y = digits.data, digits.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_train_one_hot = np.eye(10)[y_train]
y_test_one_hot = np.eye(10)[y_test]

class NeuralNetwork:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size

        self.W1 = np.random.randn(self.input_size, self.hidden_size1) * np.sqrt(2. / self.input_size)
        self.W2 = np.random.randn(self.hidden_size1, self.hidden_size2) * np.sqrt(2. / self.hidden_size1)
        self.W3 = np.random.randn(self.hidden_size2, self.output_size) * np.sqrt(2. / self.hidden_size2)

        self.b1 = np.zeros((1, self.hidden_size1))
        
        self.b2 = np.zeros((1, self.hidden_size2))
        
        self.b3 = np.zeros((1, self.output_size))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.relu(self.z2)
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = self.sigmoid(self.z3)
        return self.a3
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def backward(self, X, y, output, learning_rate):
        self.output_error = y - output
        self.output_delta = self.output_error * self.sigmoid_derivative(output)
        
        self.z2_error = self.output_delta.dot(self.W3.T)
        self.z2_delta = self.z2_error * self.relu_derivative(self.a2)
        
        self.z1_error = self.z2_delta.dot(self.W2.T)
        self.z1_delta = self.z1_error * self.relu_derivative(self.a1)
        
        self.W3 += self.a2.T.dot(self.output_delta) * learning_rate
        self.b3 += np.sum(self.output_delta, axis=0, keepdims=True) * learning_rate
        self.W2 += self.a1.T.dot(self.z2_delta) * learning_rate
        self.b2 += np.sum(self.z2_delta, axis=0, keepdims=True) * learning_rate
        self.W1 += X.T.dot(self.z1_delta) * learning_rate
        self.b1 += np.sum(self.z1_delta, axis=0, keepdims=True) * learning_rate
    
    def train(self, X, y, epochs, learning_rate):
        for i in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output, learning_rate)
            if i % 100 == 0:
                loss = np.mean(np.abs(self.output_error))
                print(f'Epoch {i}/{epochs} - Loss: {loss}')
    
    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

input_size = 64
hidden_size1 = 40
hidden_size2 = 20
output_size = 10
learning_rate = 0.001
epochs = 1000

nn = NeuralNetwork(input_size, hidden_size1, hidden_size2, output_size)
nn.train(X_train, y_train_one_hot, epochs, learning_rate)

predictions = nn.predict(X_test)
accuracy = np.mean(predictions == y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')

def show_prediction(nn, X_test, y_test, index):
    image = X_test[index].reshape(8, 8)
    true_label = y_test[index]
    
    prediction = nn.predict(X_test[index].reshape(1, -1))
    
    plt.imshow(image, cmap='gray')
    plt.title(f'True Label: {true_label}, Predicted Label: {prediction[0]}')
    plt.show()

show_prediction(nn, X_test, y_test, index=52)