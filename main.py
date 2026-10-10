import numpy as np
from tensorflowimplementation import TensorFlowClassifier
import Dataset

X_train = np.asarray(Dataset.X_train, dtype=np.float32)
y_train = np.asarray(Dataset.y_train, dtype=np.float32)

X_test = np.asarray(Dataset.X_test, dtype=np.float32)
y_test = np.asarray(Dataset.y_test, dtype=np.int32)

classifier = TensorFlowClassifier(
    learning_rate=0.1,
    epochs=100
)

classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)
accuracy = np.mean(predictions == y_test)

print(f"Test accuracy: {accuracy * 100:.2f}%")

weights, bias = classifier.get_parameters()

print("Weights:", weights)
print("Bias:", bias)

np.savez(
    "tensorflow_params.npz",
    weights=weights,
    bias=bias
)