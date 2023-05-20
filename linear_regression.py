from __future__ import annotations
from typing import Self
import numpy as np
import sys

"""
N = No of features
M = No of rows
time complexity: O(N ^ 2 * M)
space complexity = O(size of the dataset)
"""


SEED: int = 42
SIZE: int = 100
LEARNING_RATE: float = 0.1   # learning rate
EPOCHS: int = 1000


class LinearRegression(object):
    def __init__(self, *, eta: float = LEARNING_RATE, epochs: int = EPOCHS) -> None:
        self.LEARNING_RATE = eta
        self.EPOCHS = epochs

    def fit(self, X_data: np.ndarray, y_data: np.ndarray) -> Self:
        self.X_data = X_data
        self.y_data = y_data
        self.size = len(X_data[0]) + 1
        self.weights = np.random.randn(self.size)   # intialze with some random weights
        self._gradient_descent()
        return self
    
    @property
    def weights_(self) -> np.ndarray[float]:
        return self.weights.copy()
    
    def _partial_derivative(self, weight_index: int) -> float:
        root_mean_squared_error = self._cost_function()
        derivative = 0

        for index, row in enumerate(self.X_data):
            term = self.weights[0]
            CO_EFFICIENT = 1 if weight_index == 0 else row[weight_index - 1]

            for i, x in enumerate(row, start=1):
                term += self.weights[i] * x

            term -= self.y_data[index] 
            term *= CO_EFFICIENT
            derivative += term

        return (derivative / len(self.y_data)) / root_mean_squared_error

    def _gradient_descent(self) -> None:
        for epoch in range(self.EPOCHS):
            self.weights = self.weights - self.LEARNING_RATE * self._gradient()
            print(f"{epoch=}, {self.weights=}")

        """
        # another approach using keyboard interupt for unlimited epochs
        best_weights = None
        best_cost = sys.maxsize
        epoch = 1

        try:
            while True:
                self.weights = self.weights - self.LEARNING_RATE * self._gradient()
                cost = self._cost_function()

                if cost < best_cost:
                    best_cost = cost
                    best_weights = self.weights
                
                print(f"{epoch=}, and {cost=}")
                epoch += 1
        except KeyboardInterrupt:
            print(f"best cost function value is {best_cost}, and the weights are {best_weights}")
        """

    def _gradient(self) -> np.ndarray[float]:
        gradient = np.zeros(self.size)

        for weight_index, _ in enumerate(gradient):
            gradient[weight_index] = self._partial_derivative(weight_index)

        return gradient

    def _cost_function(self) -> float:
        mean_squared_error = 0

        for index, row in enumerate(self.X_data):   # iterate over all the rows in the dataset
            term = self.weights[0]

            for i, x in enumerate(row[1:], start=1):    # iterate over all the columns in the row
                term += self.weights[i] * x

            mean_squared_error += (term - self.y_data[index]) ** 2

        mean_squared_error = mean_squared_error / len(self.y_data)

        return mean_squared_error ** 0.5
    

def generate_random_data(SEED: int) -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(SEED)

    X_data = 2 * np.random.rand(SIZE, 1)    # with only one variable
    y_data = 4 + 3 * X_data + np.random.randn(SIZE, 1)

    return X_data, y_data


def main() -> int:
    X_data, y_data = generate_random_data(SEED)

    linear_regression = LinearRegression(eta=LEARNING_RATE, epochs=EPOCHS)
    linear_regression.fit(X_data, y_data)

    print(linear_regression.weights_)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
