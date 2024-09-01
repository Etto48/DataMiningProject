from __future__ import annotations
from dmml_project.dataset import Dataset

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
    
        