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

class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.sequential = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.sequential(out[:, -1, :])
        return out

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
            "relief",
            "sadness",
            "surprise",
            "worry",
        ]
        self.classes_ = sorted(kwargs.get("classes", self.classes_))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = self.params.get("max_len", 140)
        base_size = self.params.get("base_size", 128)
        depth = self.params.get("depth", 3)
        dropout = self.params.get("dropout", 0)
        match encoding:
            case "tfidf" | "count" | "binary":
                self.preprocessor: Preprocessor = Preprocessor.load(f"{PROJECT_ROOT}/data/preprocessor/{encoding}.pkl")
                self.model = nn.Sequential().to(self.device)
                for i in range(depth):
                    in_size = len(self.preprocessor) if i == 0 else base_size * 2 ** (depth-i)
                    out_size = base_size * 2 ** (depth-i-1) if i < depth-1 else len(self.classes_)
                    if dropout > 0:
                        self.model.add_module(f"dropout_{i}", nn.Dropout(dropout).to(self.device))
                    self.model.add_module(f"linear_{i}", \
                        nn.Linear(in_size, out_size).to(self.device))
                    if i < depth-1:
                        self.model.add_module(f"relu_{i}", nn.ReLU().to(self.device))
                    else:
                        self.model.add_module(f"softmax", nn.Softmax(dim=1).to(self.device))
                
            case "embeddings":
                self.preprocessor = Preprocessor.load(f"{PROJECT_ROOT}/data/preprocessor/binary.pkl")
                self.model = nn.Sequential(
                    nn.Embedding(len(self.preprocessor) + 2, base_size),
                    LSTM(base_size, base_size, depth, len(self.classes_), dropout)
                ).to(self.device)

            case _:
                raise ValueError(f"Unknown encoding: {encoding}")
        
    def train(self, train: Dataset, **kwargs):
        weight = train.class_weights()
        weight = [weight[label] for label in self.classes_]
        weight = torch.tensor(weight, dtype=torch.float32, device=self.device)
        epochs = kwargs.get("epochs", self.params.get("epochs", 10))
        batch_size = kwargs.get("batch_size", self.params.get("batch_size", 32))
        lr = kwargs.get("lr", self.params.get("lr", 0.001))
        optimizer_name = kwargs.get("optimizer", self.params.get("optimizer", "adam"))
        match optimizer_name:
            case "adam":
                optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            case "sgd":
                optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
            case _:
                raise ValueError(f"Unknown optimizer: {optimizer_name}")
        history_train = []
        history_valid = []
        metric = kwargs.get("metric", accuracy_score)
        loss_fn = nn.CrossEntropyLoss(weight=weight).to(self.device)
        for epoch in range(epochs):
            batch_count = len(train) // batch_size + (1 if len(train) % batch_size != 0 else 0)
            y_true_all = []
            y_pred_all = []
            for batch_index in tqdm(range(batch_count), desc=f"Epoch {epoch+1}/{epochs}"):
                batch = train.batch(batch_index, batch_size)
                x = batch.get_x()
                y = batch.get_y()
                y_true_all.extend(y)
                y = torch.tensor([self.classes_.index(label) for label in y], device=self.device)
                optimizer.zero_grad()
                y_pred = self._predict(x)
                y_pred_indices = torch.argmax(y_pred, dim=1).cpu().numpy()
                y_pred_all.extend([self.classes_[index] for index in y_pred_indices])
                loss = loss_fn(y_pred, y)
                loss.backward()
                optimizer.step()
            history_train.append(metric(y_true_all, y_pred_all))
            if "valid" in kwargs:
                valid = kwargs["valid"]
                assert isinstance(valid, Dataset), "Validation dataset must be provided as a Dataset object"
                metric = kwargs.get("metric", accuracy_score)
                true_y = valid.get_y()
                predicted_y = self.predict(valid.get_x())
                score = metric(true_y, predicted_y)
                history_valid.append(score)
                
        if "valid" in kwargs:
            return history_train, history_valid
        else:
            return history_train
                
    def _predict(self, x: list[str], **kwargs) -> torch.Tensor:
        if self.params.get("encoding") == "embeddings":
            x = self.preprocessor.get_indices(x, pad_to=self.max_len)
            x = torch.tensor(x, dtype=torch.long, device=self.device)
        else:
            x = self.preprocessor(x).todense()
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        y_pred = self.model(x)
        return y_pred
    
    def predict(self, x: list[str] | str, **kwargs) -> np.ndarray | str:
        with torch.no_grad():
            single_mode = isinstance(x, str)
            if single_mode:
                x = [x]
            batch_size = kwargs.get("batch_size", self.params.get("batch_size", 32))
            y_pred = []
            for i in tqdm(range(0, len(x), batch_size), "Predicting"):
                start_index = i
                end_index = min(i + batch_size, len(x))
                x_batch = x[start_index:end_index]
                y_pred_batch = self._predict(x_batch, **kwargs)
                y_pred.append(y_pred_batch)
            y_pred = torch.cat(y_pred)
            labels = torch.argmax(y_pred, dim=1).cpu()
            str_labels = [self.classes_[label] for label in labels]
            return str_labels[0] if single_mode else str_labels
    def reset(self):
        self.__init__(**self.params)
    def save(self, path: str):
        raise NotImplementedError
    def load(self, path: str):
        raise NotImplementedError
    def classes(self) -> list[str]:
        return self.classes_