import tensorflow as tf


class TensorFlowClassifier:
    def __init__(self, learning_rate=0.1, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model = None
        self.history = None

    def build_model(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(2,)),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])

        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(
                learning_rate=self.learning_rate
            ),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

    def fit(self, X, y):
        self.build_model()

        self.history = self.model.fit(
            X, y,
            epochs=self.epochs,
            verbose=0
        )

        return self

    def predict(self, X):
        probabilities = self.model.predict(X, verbose=0)
        return (probabilities >= 0.5).astype(int).ravel()

    def get_parameters(self):
        weights, bias = self.model.layers[0].get_weights()
        return weights.ravel(), float(bias[0])