"""
Implementing Logistic Regression in python using numpy and scikit-learn for metrics and data, DataFrame from pandas
"""
from __future__ import annotations
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import numpy as np
from math import log
from typing import (
    Self,
    Final,
)


ALPHA: Final[float] = 0.2    # learning rate, here it is called alpha
SEED: Final[int] = 42
EPOCHS: Final[int] = 1000
THRESHOLD: Final[float] = 0.5


class LogisticRegression(object):
    def __init__(self, alpha: float = ALPHA, threshold: float = THRESHOLD, epochs: int = EPOCHS) -> None:
        self.ALPHA = alpha
        self.EPOCHS = epochs
        self.THRESHOLD = threshold

    def fit(self, X_data: np.ndarray, y_data: np.ndarray) -> Self:
        self.X_data = X_data
        self.y_data = y_data
        self.NUMBER_OF_WEIGHTS = len(X_data[0]) + 1
        self.weights = np.random.randn(self.NUMBER_OF_WEIGHTS)
        self._gradient_descent()
        return self

    @property
    def weights_(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        return self.weights.copy()

    def _gradient_descent(self) -> None:
        for epoch in range(self.EPOCHS):
            self.weights = self.weights - self.ALPHA * self._gradient()
            # print(f"{epoch=}, cost = {self._cost_function()}")

    def predict_probability(self, x_vector: np.ndarray) -> float:
        probability = self.weights[0]

        for index, x in enumerate(x_vector, start=1):
            probability += self.weights[index] * x

        return (probability := self.sigmoid(probability))

    def predict_class(self, x_vector: np.ndarray) -> int:
        probability = self.predict_probability(x_vector)
        return 1 if probability >= self.THRESHOLD else 0

    def _partial_derivative(self, weight_index: int) -> float:
        derivative = 0

        for index, row in enumerate(self.X_data):
            CO_EFFICIENT = 1 if weight_index == 0 else row[weight_index - 1]
            term = self.weights[0]

            for i, x in enumerate(row, start=1):
                term += self.weights[i] * x

            term = self.sigmoid(term)
            term -= self.y_data[index]
            term *= CO_EFFICIENT
            derivative += term

        return derivative / len(self.y_data)

    def _gradient(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        return np.array([self._partial_derivative(weight_index) for weight_index in range(self.NUMBER_OF_WEIGHTS)])

    def _cost_function(self) -> float:      # this is the log loss cost function
        log_loss = 0.0

        for index, row in enumerate(self.X_data):
            probability = self.predict_probability(row)
            log_loss += self.y_data[index] * log(probability) + (1 - self.y_data[index]) * log(1 - probability)

        log_loss = -1 / len(self.y_data) * log_loss

        return log_loss

    @staticmethod
    def sigmoid(x: float) -> float:
        return 1 / (1 + (np.exp(-x)))


def generate_data(seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    if seed is None:
        seed = 42

    np.random.seed(seed)
    dataset = load_iris(as_frame=True)
    X_data = dataset.data[["petal width (cm)", "sepal length (cm)", "sepal width (cm)", "petal length (cm)"]].values
    y_data = dataset.target_names[dataset.target] == "virginica"
    X_data = np.array(X_data)
    y_data = np.array(y_data, dtype=np.int16)   # convert the boolean values to int

    return X_data.copy(), y_data.copy()


def main() -> int:
    for seed in range(42, 50):
        X_data, y_data = generate_data(seed)
        X_data_train, X_data_test, y_data_train, y_data_test = train_test_split(X_data, y_data, random_state=42)

        logistic_regression = LogisticRegression(alpha=ALPHA, threshold=THRESHOLD, epochs=EPOCHS)
        logistic_regression = logistic_regression.fit(X_data_train, y_data_train)

        print(f"confusion matrix = {confusion_matrix(y_data_test, [logistic_regression.predict_class(row) for row in X_data_test])}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
