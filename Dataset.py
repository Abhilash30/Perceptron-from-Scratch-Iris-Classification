from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris = load_iris()

X = iris.data[:100, 2:4]

# Setosa = 0, Versicolor = 1
y = iris.target[:100]

X_train, X_test, y_train, y_test = train_test_split(
X, y,
test_size=0.2,
random_state=42,
stratify=y
)