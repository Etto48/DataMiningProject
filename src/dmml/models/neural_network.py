from dmml.dataset import Dataset
from dmml.models.model import Model


class NeuralNetwork(Model):
    def train(self, train: Dataset, **kwargs):
        raise NotImplementedError
    def evaluate(self, valid: Dataset, **kwargs):
        raise NotImplementedError
    def predict(self, x: str, **kwargs) -> str:
        raise NotImplementedError
    def save(self, path: str):
        raise NotImplementedError
    def load(self, path: str):
        raise NotImplementedError