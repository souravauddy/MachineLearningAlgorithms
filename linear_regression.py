from __future__ import annotations
import numpy as np
import sys
from typing import (
    Self,
    Final,
)
from numpy.typing import (
    NDArray,
)


"""
N = No of features
M = No of rows
time complexity: O(N ^ 2 * M)
space complexity = O(size of the dataset)
"""


SEED: Final[int] = 42
SIZE: Final[int] = 100
LEARNING_RATE: Final[float] = 0.1   # learning rate
EPOCHS: Final[int] = 1000


class LinearRegression(object):
    def __init__(self, *, eta: float = LEARNING_RATE, epochs: int | None = EPOCHS) -> None:
        """
        takes input the hyperparameters learning rate and the number of epochs

        Args:
            eta (float, optional): learning rate given as float. Defaults to LEARNING_RATE.
            epochs (int | None, optional): no of epochs given as an integer, if None is given then unlimited no of epochs is performed, stops using Keyboard Interrupt. Defaults to EPOCHS.
        """

        self.LEARNING_RATE = eta
        self.EPOCHS = epochs

    def fit(self, X_data: NDArray, y_data: NDArray) -> Self:
        """
        Uses gradient descent to find the optimal weights, and returns and instance of itself.

        Args:
            X_data (NDArray): data of the input features
            y_data (NDArray): data of the target features

        Returns:
            Self: returns an instance of itself after computing the weights for the sake of compatibility with sklearn.
        """

        self.X_data = X_data
        self.y_data = y_data
        self.size = len(X_data[0]) + 1
        self.weights = np.random.randn(self.size)   # intialze with some random weights

        if self.EPOCHS is not None:
            self._gradient_descent()
        else:
            self._gradient_descent(unlimited_epcohs=True)

        return self

    @property
    def weights_(self) -> NDArray:
        """returns the optimal weights, should be called after calling the fit method.

        Returns:
            NDArray: returns the weights as a numpy array.
        """

        return self.weights.copy()

    def _partial_derivative(self, weight_index: int) -> float:
        """computes the partial derivative of a single vector and returns its value.

        Args:
            weight_index (int): The index of the vector whose derivative is to be computed.

        Returns:
            float: computed parial derivative.
        """

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

    def _gradient_descent(self, unlimited_epcohs: bool = False) -> None:
        """performs gradient descent.

        Args:
            unlimited_epcohs (bool, optional): for unlimited epcohs, using keyboard interrupt. Defaults to False.
        """

        if not unlimited_epcohs:
            assert self.EPOCHS is not None, print("self.EPOCHS is None", file=sys.stderr)

            for epoch in range(self.EPOCHS):
                self.weights = self.weights - self.LEARNING_RATE * self._gradient()
                print(f"{epoch=}, {self.weights=}")

            return

        best_weights = None
        best_cost = float(sys.maxsize)
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
            assert best_weights is not None
            print(f"best cost function value is {best_cost}, and the weights are {best_weights}")

    def _gradient(self) -> NDArray:
        """returns the gradient vector, with all the computed parital derivatives.

        Returns:
            np.ndarray[tuple[int], np.dtype[np.float64]]: gradient vector.
        """

        gradient = np.zeros(self.size)

        for weight_index, _ in enumerate(gradient):
            gradient[weight_index] = self._partial_derivative(weight_index)

        return gradient

    def _cost_function(self) -> float:
        """Cost function for calculating the loss. Here, we used Root Mean Squared Error.

        Returns:
            float: cost
        """

        mean_squared_error = 0.0

        for index, row in enumerate(self.X_data):   # iterate over all the rows in the dataset
            term = self.weights[0]

            for i, x in enumerate(row[1:], start=1):    # iterate over all the columns in the row
                term += self.weights[i] * x

            mean_squared_error += (term - self.y_data[index]) ** 2

        mean_squared_error = mean_squared_error / len(self.y_data)

        return mean_squared_error ** 0.5


def generate_random_data(seed: int) -> tuple[NDArray, NDArray]:
    """generates random data for linear regression, using the SEED parameter.

    Args:
        seed (int): value to reproduce the results

    Returns:
        tuple[NDArray, NDArray]: two numpy arrays, input features and output column
    """

    np.random.seed(seed)

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
