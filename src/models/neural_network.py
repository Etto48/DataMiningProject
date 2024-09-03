from typing import Any, Callable, Literal
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from dmml_project.dataset import Dataset
from dmml_project.models import Model
from dmml_project import PROJECT_ROOT
from dmml_project.preprocessor import Preprocessor
from tqdm import tqdm
import numpy as np

class NeuralNetwork(Model):
    def __init__(self, **kwargs):
        self.params = kwargs
        encoding = kwargs.get("encoding", "tfidf")
        self.classes_ = [
            "anger",
            "boredom",
            "empty",
            "enthusiasm",
            "fun",
            "happiness",
            "hate",
            "love",
            "neutral",
            "sadness",
            "surprise",
            "worry",
        ]
        self.classes_ = sorted(kwargs.get("classes", self.classes_))
        match encoding:
            case "tfidf" | "count" | "binary":
                self.preprocessor: Preprocessor = Preprocessor.load(f"{PROJECT_ROOT}/data/preprocessor/{encoding}.pkl")
                self.model = torch.nn.Sequential(
                    nn.Linear(len(self.preprocessor), 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, len(self.classes_)),
                    nn.Softmax(dim=1),
                )
            case "embeddings":
                raise NotImplementedError
            case _:
                raise ValueError(f"Unknown encoding: {encoding}")
        
    def train(self, train: Dataset, **kwargs):
        epochs = kwargs.get("epochs", self.params.get("epochs", 10))
        batch_size = kwargs.get("batch_size", self.params.get("batch_size", 32))
        optimizer_name = kwargs.get("optimizer", self.params.get("optimizer", "adam"))
        match optimizer_name:
            case "adam":
                optimizer = torch.optim.Adam(self.model.parameters())
            case "sgd":
                optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
            case _:
                raise ValueError(f"Unknown optimizer: {optimizer_name}")
        loss_fn = nn.CrossEntropyLoss()
        for epoch in tqdm(range(epochs), "Epochs"):
            batch_index = 0
            for i in tqdm(range(0, len(train), batch_size), "Batches"):
                x, y = train.get_batch(batch_index, batch_size)
                x = self.preprocessor(x)
                x = torch.tensor(x, dtype=torch.float32)
                y = torch.tensor([self.classes_.index(label) for label in y])
                optimizer.zero_grad()
                y_pred = self.model(x)
                loss = loss_fn(y_pred, y)
                loss.backward()
                optimizer.step()
                batch_index += 1
                
    def predict(self, x: list[str] | str, **kwargs) -> np.ndarray | int:
        if isinstance(x, str):
            x = [x]
        x = self.preprocessor(x)
        x = torch.tensor(x, dtype=torch.float32)
        y_pred = self.model(x)
        labels = torch.argmax(y_pred, dim=1)
        return labels[0].item() if isinstance(x, str) else labels.numpy()
    def reset(self):
        self.__init__(**self.params)
    def save(self, path: str):
        raise NotImplementedError
    def load(self, path: str):
        raise NotImplementedError
    def classes(self) -> list[str]:
        return self.classes_