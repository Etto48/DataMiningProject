from __future__ import annotations
from dmml_project.dataset import Dataset
from tqdm import tqdm

class Model:
    def train(self, train: Dataset, **kwargs):
        raise NotImplementedError
    def evaluate(self, valid: Dataset, **kwargs) -> float:
        raise NotImplementedError
    def predict(self, x: str, **kwargs) -> str:
        raise NotImplementedError
    def save(self, path: str):
        raise NotImplementedError
    def load(self, path: str):
        raise NotImplementedError
    def xval(self, dataset: Dataset, folds: int = 10, **kwargs) -> list[float]:
        """
        Perform cross validation training and evaluation on the dataset.
        """
        evaluations: list[float] = []
        for i in tqdm(range(folds), "Cross validation"):
            train, valid = dataset.fold(i, folds)
            self.train(train, **kwargs)
            evaluations.append(self.evaluate(valid))
        return evaluations
    
        