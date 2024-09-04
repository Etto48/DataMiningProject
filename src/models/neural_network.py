from typing import Any, Callable, Literal
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from dmml_project.dataset import Dataset
from dmml_project.models import Model
from dmml_project import PROJECT_ROOT, CLASSES
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
    
class CNN(nn.Module):
    def __init__(self, embedding_size: int, classes:int, num_layers: int, dropout: int, batchnorm: bool):
        super().__init__()
        self.sequential = nn.Sequential()
        for i in range(num_layers):
            if dropout > 0:
                self.sequential.add_module(f"dropout_{i}", nn.Dropout(dropout))
            self.sequential.add_module(
                f"conv_{i}",
                nn.Conv1d(embedding_size, embedding_size, 3, padding=1)
            )
            if batchnorm:
                self.sequential.add_module(f"batchnorm_{i}", nn.BatchNorm1d(embedding_size))
            self.sequential.add_module(f"relu_{i}", nn.ReLU())
            self.sequential.add_module(f"maxpool_{i}", nn.MaxPool1d(2))
        self.sequential.add_module("avgpool", nn.AdaptiveAvgPool1d(1))
        self.sequential.add_module("flatten", nn.Flatten())
        self.sequential.add_module("linear", nn.Linear(embedding_size, classes))
        self.sequential.add_module("softmax", nn.Softmax(dim=1))
        
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 1)
        return self.sequential(x)

class FFNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int, dropout: float, batchnorm: bool):
        super().__init__()
        self.sequential = nn.Sequential()
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            out_size = output_size if i == num_layers-1 else hidden_size
            if dropout > 0:
                self.sequential.add_module(f"dropout_{i}", nn.Dropout(dropout))
            self.sequential.add_module(f"linear_{i}", nn.Linear(in_size, out_size))
            if batchnorm:
                self.sequential.add_module(f"batchnorm_{i}", nn.BatchNorm1d(out_size))
            if i < num_layers-1:
                self.sequential.add_module(f"relu_{i}", nn.ReLU())
            else:
                self.sequential.add_module(f"softmax", nn.Softmax(dim=1))
    def forward(self, x):
        return self.sequential(x)

class NeuralNetwork(Model):
    def __init__(self, **kwargs):
        kwargs["network"] = kwargs.get("network", "ff_tfidf")
        self.params = kwargs
        network = kwargs["network"]
        self.classes_ = CLASSES
        self.classes_ = sorted(kwargs.get("classes", self.classes_))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = self.params.get("max_len", 140)
        base_size = self.params.get("base_size", 128)
        depth = self.params.get("depth", 3)
        dropout = self.params.get("dropout", 0)
        batchnorm = self.params.get("batchnorm", False)
        match network:
            case "ff_tfidf" | "ff_count" | "ff_binary":
                preprocessor = network.split("_")[1]
                self.preprocessor: Preprocessor = Preprocessor.load(f"{PROJECT_ROOT}/data/preprocessor/{preprocessor}.pkl")
                self.model = FFNN(len(self.preprocessor), base_size, len(self.classes_), depth, dropout, batchnorm).to(self.device)
                
            case "cnn_embeddings" | "lstm_embeddings":
                self.preprocessor = Preprocessor.load(f"{PROJECT_ROOT}/data/preprocessor/binary.pkl")
                if network == "lstm_embeddings":
                    self.model = nn.Sequential(
                        nn.Embedding(len(self.preprocessor) + 2, base_size),
                        LSTM(base_size, base_size, depth, len(self.classes_), dropout)
                    ).to(self.device)
                elif network == "cnn_embeddings":
                    self.model = nn.Sequential(
                        nn.Embedding(len(self.preprocessor) + 2, base_size),
                        CNN(base_size, len(self.classes_), depth, dropout, batchnorm)
                    ).to(self.device)
                    
            case _:
                raise ValueError(f"Unknown encoding: {network}")
        
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
            self.model.train()
            for batch_index in tqdm(range(batch_count), desc=f"Epoch {epoch+1:>{len(str(epochs))}}/{epochs}"):
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
        if "embeddings" in self.params["network"]:
            x = self.preprocessor.get_indices(x, pad_to=self.max_len)
            x = torch.tensor(x, dtype=torch.long, device=self.device)
        else:
            x = self.preprocessor(x).todense()
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        y_pred = self.model(x)
        return y_pred
    
    def predict(self, x: list[str] | str, **kwargs) -> np.ndarray | str:
        with torch.no_grad():
            self.model.eval()
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
        with open(path, 'wb') as f:
            torch.save((self.params, self.model.state_dict()), f)
    
    def load(path: str) -> 'NeuralNetwork':
        with open(path, 'rb') as f:
            params, state_dict = torch.load(f)
        self = NeuralNetwork(**params)
        self.model.load_state_dict(state_dict)
        return self
    
    def classes(self) -> list[str]:
        return self.classes_
    