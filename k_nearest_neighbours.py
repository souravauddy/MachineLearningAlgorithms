from __future__ import annotations
from ast import Call
import numpy as np
import pandas as pd
import scipy.spatial.distance    
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from typing import (
    Any,
    Callable,
    Mapping,
    Optional,
    Self,
    TypeVar,
    Generic,
    TypedDict,
    Unpack,
)
from numpy.typing import (
    NDArray
)


TData = TypeVar("TData", bound=NDArray)


class TypedKeywordArguments(TypedDict, total=False):
    p: int


class KNearestNeighboursClassifier(Generic[TData]):
    def __init__(self, *, n_neighbors: int | None = None, distance_type: str = "euclidean", **_kwargs: Unpack[TypedKeywordArguments]) -> None:
        """Takes input the hyperparameters, for the KNN classifier.

        Args:
            n_neighbors (int | None, optional): no of neighbors to consider. Defaults to None.
            distance_type (str, optional): type of the distance to use. Defaults to "euclidean".
        """

        if n_neighbors is None:
            n_neighbors = 3

        self.n_neighbors = n_neighbors | 1
        self.distance_type = distance_type
        self.p = _kwargs.get('p', None)

    def fit(self, X_data: TData, y_data: TData) -> Self:
        """Takes input the data and stores it all the computation is done during the predict method call.

        Args:
            X_data (TData): vector of input features
            y_data (TData): vector of output features

        Returns:
            Self: returns the same instance, with which it was called. 
        """
        
        self.X_data = X_data
        self.y_data = y_data
        return self

    def predict(self, prediction_vector: TData) -> int:
        """Predict which class the vector belongs to.

        Args:
            prediction_vector (TData): classify the type of the prediction vector.

        Returns:
            int: returns the class of the vector. (0 or 1) in binary classification.
        """

        distance_function: function = {
            "euclidean": self.euclidean_distance,
            "manhattan": self.manhattan_distance,
            "minkowski": self.minkowski_distance,
        }.get(self.distance_type, self.euclidean_distance)

        distances = []

        for vector, data in zip(self.X_data, self.y_data):
            distance = distance_function(vector, prediction_vector, p=self.p)   #type: ignore
            distances.append((distance, data))

        distances.sort()
        neighbors = distances[:self.n_neighbors]
        ones_count = sum((neighbor[1] for neighbor in neighbors))
        zeros_count = self.n_neighbors - ones_count

        return int(ones_count > zeros_count) 
    
    @staticmethod
    def euclidean_distance(vector1: TData, vector2: TData, _: None = None) -> float:
        """Returns the euclidean distance between two vectors.

        Args:
            vector1 (TData): first vector
            vector2 (TData): second vector

        Returns:
            float: distance between the two vectors
        """

        return scipy.spatial.distance.euclidean(vector1, vector2)
    
    @staticmethod
    def manhattan_distance(vector1: TData, vector2: TData, _: None = None) -> float:
        """Returns the manhattan distance between the two vectors.

        Args:
            vector1 (TData): first vector.
            vector2 (TData): second vector.

        Returns:
            float: manhattan distance of the two vectors.
        """
        
        return scipy.spatial.distance.minkowski(vector1, vector2, p=1)
    
    @staticmethod
    def minkowski_distance(vector1: TData, vector2: TData, p: int) -> float:
        """Returns the minkowski distance between the two vectors using the 'p' hyperparameter.

        Args:
            vector1 (TData): first vector.
            vector2 (TData): second vector.
            p (int): hyperparameter for minkowski distance.

        Returns:
            float: minkowski distance between the two vectors.
        """
        
        return scipy.spatial.distance.minkowski(vector1, vector2, p=p)
    

def generate_data(seed: int | None = None) -> tuple[NDArray, NDArray]:
    """Generate random data using seed, or use builtin datasets in sklearn.

    Args:
        seed (int | None, optional): seed value to initialze random generator, to produce reproducable results. Defaults to None.

    Returns:
        tuple[NDArray, NDArray]: input data and output data.
    """
    
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
    X_data, y_data = generate_data()
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data)
    k_nearest_neighbors_classifier = KNearestNeighboursClassifier(n_neighbors=5, distance_type="minkowski", p=1)    #type: ignore
    k_nearest_neighbors_classifier = k_nearest_neighbors_classifier.fit(X_train, y_train)
    predictions = np.array([k_nearest_neighbors_classifier.predict(vector) for vector in X_test])
    score = accuracy_score(y_test, predictions)

    print(f"{score=}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
