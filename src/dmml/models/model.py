from __future__ import annotations

from dmml.dataset import Dataset
from dmml.models.decision_tree import DecisionTree
from dmml.models.hyperparameters import Hyperparameters
from dmml.models.neural_network import NeuralNetwork
from dmml.models.random_forest import RandomForest


class Model:
    def __init__(self, params: Hyperparameters) -> Model:
        if params["kind"] == "decision_tree":
            return DecisionTree(params)
        elif params["kind"] == "random_forest":
            return RandomForest(params)
        elif params["kind"] == "neural_network":
            return NeuralNetwork(params)
        else:
            raise ValueError(f"Unknown model kind '{params['kind']}'")

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
    
        