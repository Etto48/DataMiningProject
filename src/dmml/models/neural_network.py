from dmml.dataset import Dataset
from dmml.models import Hyperparameters, Model

class NeuralNetwork(Model):
    def __init__(self, params: Hyperparameters):
        raise NotImplementedError
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