from __future__ import annotations
import numpy as np
import pandas as pd     # type: ignore
import scipy.spatial    # type: ignore
from sklearn.datasets import load_iris      # type: ignore
from sklearn.model_selection import train_test_split        # type: ignore
from sklearn.metrics import accuracy_score      # type: ignore
from typing import (
    Mapping,
    Self,
    TypeVar,
    Generic,
)
from collections.abc import Mapping


TData = TypeVar("TData", np.ndarray, pd.DataFrame)


class KNearestNeighboursClassifier(Generic[TData]):
    def __init__(self, *, n_neighbors: int | None = None, distance_type: str = "euclidean", **_kwargs: Mapping[str, int]) -> None:
        if n_neighbors is None:
            n_neighbors = 3

        self.n_neighbors = n_neighbors | 1
        self.distance_type = distance_type
        self.p = _kwargs.get('p') if _kwargs is not None and isinstance(_kwargs, Mapping) else None

    def fit(self, X_data: TData, y_data: TData) -> Self:
        self.X_data = X_data
        self.y_data = y_data
        return self

    def predict(self, prediction_vector: TData) -> int:
        distance_function = {
            "euclidean": self.euclidean_distance,
            "manhattan": self.manhattan_distance,
            "minkowski": self.minkowski_distance,
        }.get(self.distance_type, self.euclidean_distance)

        distances = []

        for index, vector in enumerate(self.X_data):
            distance = distance_function(vector, prediction_vector, p=self.p)
            distances.append((distance, self.y_data[index]))

        distances.sort(
            key=type(
                "_TupleComparison",
                (tuple,),
                {
                    "__lt__": lambda x, y: x[0] < y[0]
                }
            )
        )

        neighbors = distances[:self.n_neighbors]
        ones_count = sum((neighbor[1] for neighbor in neighbors))
        zeros_count = self.n_neighbors - ones_count

        return int(ones_count > zeros_count) 
    
    @staticmethod
    def euclidean_distance(vector1: TData, vector2: TData, **_) -> float:
        return scipy.spatial.distance.euclidean(vector1, vector2)
    
    @staticmethod
    def manhattan_distance(vector1: TData, vector2: TData, **_) -> float:
        return scipy.spatial.distance.minkowski(vector1, vector2, p=1)
    
    @staticmethod
    def minkowski_distance(vector1: TData, vector2: TData, p: int) -> float:
        return scipy.spatial.distance.minkowski(vector1, vector2, p=p)
    

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
    X_data, y_data = generate_data()
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data)
    k_nearest_neighbors_classifier = KNearestNeighboursClassifier(n_neighbors=5, distance_type="minkowski", p=1)
    k_nearest_neighbors_classifier = k_nearest_neighbors_classifier.fit(X_train, y_train)
    predictions = np.array([k_nearest_neighbors_classifier.predict(vector) for vector in X_test])
    score = accuracy_score(y_test, predictions)

    print(f"{score=}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
