from __future__ import annotations
from typing import Any, Callable
from sklearn.metrics import accuracy_score
from dmml_project.dataset import Dataset
from tqdm import tqdm
import numpy as np

class Model:
    def train(self, train: Dataset, **kwargs):
        ...
    def evaluate(self, valid: Dataset, metric: Callable[[Any, Any], Any] = accuracy_score, **kwargs) -> Any:
        true_y = valid.get_y()
        predicted_y = self.predict(valid.get_x())
        return metric(true_y, predicted_y)
    def predict(self, x: list[str] | str, **kwargs) -> np.ndarray | int:
        ...
    def reset(self):
        ...
    def save(self, path: str):
        ...
    def load(self, path: str):
        ...
    def classes(self) -> list[str]:
        ...
    def xval(self, dataset: Dataset, folds: int = 10, metric: Callable[[Any, Any], Any] = accuracy_score, **kwargs) -> list[Any]:
        """
        Perform cross validation training and evaluation on the dataset.
        """
        evaluations: list = []
        for i in tqdm(range(folds), "Cross validation"):
            self.reset()
            train, valid = dataset.fold(i, folds)
            self.train(train, **kwargs)
            evaluations.append(self.evaluate(valid, metric))
        self.reset()
        return evaluations
    