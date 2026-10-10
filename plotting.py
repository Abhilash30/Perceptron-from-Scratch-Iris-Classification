import matplotlib.pyplot as plt
import numpy as np
import Dataset

X = np.asarray(Dataset.X)
y = np.asarray(Dataset.y)

plt.figure(figsize=(9, 6))

# Plot Iris setosa
plt.scatter(
    X[y == 0, 0],
    X[y == 0, 1],
    label="Iris setosa",
    marker="o",
    alpha=0.8
)

# Plot Iris versicolor
plt.scatter(
    X[y == 1, 0],
    X[y == 1, 1],
    label="Iris versicolor",
    marker="^",
    alpha=0.8
)

params = np.load("tensorflow_params.npz")
weights = params["weights"].ravel()
bias = float(params["bias"])

# Decision boundary: z = w1*x1 + w2*x2 + b = 0
x_line = np.linspace(X[:, 0].min() - 0.5,
                     X[:, 0].max() + 0.5, 200)

if abs(weights[1]) > 1e-10:
    y_line = -(weights[0] * x_line + bias) / weights[1]
    plt.plot(
        x_line,
        y_line,
        "k--",
        linewidth=2,
        label="TensorFlow decision boundary"
    )
elif abs(weights[0]) > 1e-10:
    # w2 is zero, so the boundary is a vertical line
    plt.axvline(
        x=-bias / weights[0],
        color="black",
        linestyle="--",
        linewidth=2,
        label="TensorFlow decision boundary"
    )

plt.xlabel("Petal length (cm)")
plt.ylabel("Petal width (cm)")
plt.title("Iris Classification — Perceptron Dataset")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("iris_plot_tensorflow.png", dpi=150)
plt.close()
