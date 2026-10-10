import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.weights = np.zeros(X.shape[1])
        self.bias = 0.0

        for _ in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                y_hat = self._predict_one(xi)
                update = self.lr * (target - y_hat) 
                self.weights += update * xi
                self.bias += update
                errors += int(update != 0)
            if errors == 0:  # converged
                break
        return self

    def _predict_one(self, x):
        z = np.dot(self.weights, x) + self.bias
        return 1 if z >= 0 else 0

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        z = X @ self.weights + self.bias
        return np.where(z >= 0, 1, 0)

    
