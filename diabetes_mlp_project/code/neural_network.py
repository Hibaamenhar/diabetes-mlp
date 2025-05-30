
import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []

        np.random.seed(42)
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1)
            self.biases.append(np.zeros((1, layer_sizes[i+1])))

    def forward(self, X):
        self.activations = [X]
        self.z_values = []

        for i in range(len(self.weights) - 1):
            z = self.activations[-1] @ self.weights[i] + self.biases[i]
            self.z_values.append(z)
            a = relu(z)
            self.activations.append(a)

        z = self.activations[-1] @ self.weights[-1] + self.biases[-1]
        self.z_values.append(z)
        output = sigmoid(z)
        self.activations.append(output)

        return output

    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        loss = -np.mean(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))
        return loss

    def backward(self, y_true):
        m = y_true.shape[0]
        dZ = self.activations[-1] - y_true
        self.d_weights = []
        self.d_biases = []

        for i in reversed(range(len(self.weights))):
            a_prev = self.activations[i]
            dW = a_prev.T @ dZ / m
            db = np.sum(dZ, axis=0, keepdims=True) / m
            self.d_weights.insert(0, dW)
            self.d_biases.insert(0, db)

            if i > 0:
                dZ = (dZ @ self.weights[i].T) * relu_derivative(self.z_values[i - 1])

    def update_weights(self):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * self.d_weights[i]
            self.biases[i] -= self.learning_rate * self.d_biases[i]

    def train(self, X, y, epochs=100, batch_size=32):
        history = []
        for epoch in range(epochs):
            indices = np.random.permutation(X.shape[0])
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                outputs = self.forward(X_batch)
                self.backward(y_batch)
                self.update_weights()

            outputs = self.forward(X)
            loss = self.compute_loss(y, outputs)
            history.append(loss)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        return history

    def predict(self, X):
        outputs = self.forward(X)
        return (outputs >= 0.5).astype(int)
