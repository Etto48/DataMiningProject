import torch
#import torchtext
import torch.nn as nn
from dmml_project.metrics import f1_score
from dmml_project.dataset import Dataset
from dmml_project.models import Model
from dmml_project import PROJECT_ROOT, CLASSES, EXCLUDE_REGEX
from dmml_project.preprocessor import Preprocessor
from tqdm import tqdm
import numpy as np

class TNN(nn.Module):
    def __init__(self, input_size: int, n_heads: int, hidden_size: int, output_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=input_size,
                nhead=n_heads,
                dim_feedforward=hidden_size,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers,
            norm=nn.LayerNorm(input_size)
        )
        self.sequential = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, output_size),
        )
        
    def forward(self, x, mask=None):
        out = self.transformer(x, src_key_padding_mask=~mask)
        out = out[:, 0, :]
        out = self.sequential(out)
        return out

class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        lstm_dropout = 0 if num_layers == 1 else dropout
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=lstm_dropout)
        self.sequential = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.sequential(out)
        return out
    
class CNN(nn.Module):
    def __init__(self, embedding_size: int, hidden_size: int, classes:int, num_layers: int, dropout: int, batchnorm: bool):
        super().__init__()
        self.sequential = nn.Sequential()
        for i in range(num_layers):
            if dropout > 0:
                self.sequential.add_module(f"dropout_{i}", nn.Dropout(dropout))
            in_size = embedding_size if i == 0 else hidden_size
            out_size = hidden_size
            self.sequential.add_module(
                f"pad_{i}",
                nn.ZeroPad1d((0, 2))
            )
            self.sequential.add_module(
                f"conv_{i}",
                nn.Conv1d(in_size, out_size, 3)
            )
            if batchnorm:
                self.sequential.add_module(f"batchnorm_{i}", nn.BatchNorm1d(out_size))
            self.sequential.add_module(f"relu_{i}", nn.ReLU())
            #self.sequential.add_module(f"maxpool_{i}", nn.MaxPool1d(2))
        self.sequential.add_module("avgpool", nn.AdaptiveAvgPool1d(1))
        self.sequential.add_module("flatten", nn.Flatten())
        self.sequential.add_module("linear", nn.Linear(hidden_size, classes))
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

class GloVePreprocessor:
    def __init__(self, device: torch.device):
        #self.glove = torchtext.vocab.GloVe(name="6B", dim=50, cache=f"{PROJECT_ROOT}/data/glove")
        self.glove = {}
        try:
            glove_path = f"{PROJECT_ROOT}/data/glove/glove.6B.50d.txt"
            with open(glove_path, "r", encoding="utf-8") as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    vector = np.array(values[1:], dtype=np.float32)
                    self.glove[word] = vector
        except FileNotFoundError:
            print(f"GloVe vectors not found. Make sure you have downloaded the GloVe vectors inside {glove_path}")
            exit(1)
        self.device = device
        self.regex = EXCLUDE_REGEX
    def get_vecs_by_tokens(self, tokens: list[str]) -> torch.Tensor:
        vecs = np.array([x for x in [self.glove.get(token, self.glove.get(token.lower(), None)) for token in tokens] if x is not None])
        if len(vecs) == 0:
            vecs = np.zeros((1, 50), dtype=np.float32)
        vecs = torch.tensor(vecs, dtype=torch.float32, device=self.device)
        return vecs
    def fit(self, x: list[str], verbose: bool = True):
        pass
    def __call__(self, x: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        x = [self.regex.sub(" ", sentence) for sentence in x]
        x = [self.get_vecs_by_tokens(sentence.split()) for sentence in x]
        lengths = [sentence.size(0) for sentence in x]
        mask = torch.arange(max(lengths), device=self.device).expand(len(lengths), max(lengths)) < torch.tensor(lengths, device=self.device).unsqueeze(1)
        max_len = max(lengths)
        x = [torch.nn.functional.pad(sentence, (0, 0, 0, max_len - sentence.size(0))) for sentence in x]
        x = torch.stack(x)
        return x, mask

class NeuralNetwork(Model):
    def __init__(self, **kwargs):
        super().__init__(kwargs, network="ff_tfidf")
        network = kwargs["network"]
        self.classes_ = CLASSES
        self.classes_ = sorted(kwargs.get("classes", self.classes_))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_size = self.params.get("base_size", 128)
        depth = self.params.get("depth", 3)
        dropout = self.params.get("dropout", 0)
        batchnorm = self.params.get("batchnorm", False)
        # disable dropout if batchnorm is enabled and the network supports it
        if batchnorm and "lstm" not in network:
            dropout = 0
        match network:
            case "ff_tfidf" | "ff_count" | "ff_binary":
                preprocessor = network.split("_")[1]
                self.preprocessor: Preprocessor = Preprocessor(kind=preprocessor)
                self.init_model = lambda vocab_size: FFNN(
                    vocab_size, 
                    base_size, 
                    len(self.classes_), 
                    depth, 
                    dropout, 
                    batchnorm
                ).to(self.device)
                self.model = None
                
            case "cnn_embeddings" | "lstm_embeddings":
                self.preprocessor = Preprocessor(kind="binary")
                if network == "lstm_embeddings":
                    self.init_model = lambda vocab_size: nn.Sequential(
                        nn.Embedding(vocab_size + 2, base_size),
                        LSTM(base_size, base_size, depth, len(self.classes_), dropout)
                    ).to(self.device)
                elif network == "cnn_embeddings":
                    self.init_model = lambda vocab_size: nn.Sequential(
                        nn.Embedding(vocab_size + 2, base_size),
                        CNN(base_size, base_size, len(self.classes_), depth, dropout, batchnorm)
                    ).to(self.device)
                self.model = None
            case "cnn_glove" | "lstm_glove" | "tnn_glove":
                self.preprocessor = GloVePreprocessor(self.device)
                if network == "cnn_glove":
                    self.model = CNN(50, base_size, len(self.classes_), depth, dropout, batchnorm).to(self.device)
                elif network == "lstm_glove":
                    self.model = LSTM(50, base_size, depth, len(self.classes_), dropout).to(self.device)
                elif network == "tnn_glove":
                    self.model = TNN(50, 2, base_size, len(self.classes_), depth, dropout).to(self.device)
            case _:
                raise ValueError(f"Unknown encoding: {network}")
        
    def train(self, train: Dataset, **kwargs):
        self.preprocessor.fit(train.get_x(), verbose=kwargs.get("verbose", True))
        if self.model is None:
            self.model = self.init_model(len(self.preprocessor))
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
        metric = kwargs.get("metric", f1_score)
        loss_fn = nn.CrossEntropyLoss().to(self.device)
        disable_tqdm = not kwargs.get("verbose", True)
        patience = kwargs.get("patience", self.params.get("patience", None))
        if patience != None and "valid" not in kwargs:
            print("To use early stopping you need a validation set.")
        best_model_weights = None
        best_model_accuracy = 0
        epochs_without_improvement = 0
        for epoch in range(epochs):
            sampled_train = train.random_sample(len(train))
            batch_count = len(sampled_train) // batch_size + (1 if len(sampled_train) % batch_size != 0 else 0)
            y_true_all = []
            y_pred_all = []
            self.model.train()
            batch_range = tqdm(range(batch_count), desc=f"Epoch {epoch+1:>{len(str(epochs))}}/{epochs}", disable=disable_tqdm)
            for batch_index in batch_range:
                batch = sampled_train.batch(batch_index, batch_size)
                x = batch.get_x()
                y = batch.get_y()
                y_true_all.extend(y)
                y = torch.tensor([self.classes_.index(label) for label in y], device=self.device)
                optimizer.zero_grad()
                y_pred = self._predict(x)
                y_pred_indices = torch.argmax(y_pred, dim=1).cpu().numpy()
                y_pred_all.extend([self.classes_[index] for index in y_pred_indices])
                loss = loss_fn(y_pred, y)
                batch_range.set_postfix(loss=loss.item())
                loss.backward()
                optimizer.step()
            train_score = metric(y_true_all, y_pred_all)
            history_train.append(train_score)
            if "valid" in kwargs:
                valid = kwargs["valid"]
                assert isinstance(valid, Dataset), "Validation dataset must be provided as a Dataset object"
                metric = kwargs.get("metric", f1_score)
                true_y = valid.get_y()
                
                predicted_y = self.predict(valid.get_x(), y=valid.get_y(), verbose=kwargs.get("verbose", True))
                valid_score = metric(true_y, predicted_y)
                if valid_score > best_model_accuracy + 1e-5:
                    best_model_accuracy = valid_score
                    best_model_weights = self.model.state_dict()
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                if kwargs.get("verbose", True):
                    print(f"Training: {train_score} Validation: {valid_score}")
                history_valid.append(valid_score)
            if patience is not None and epochs_without_improvement >= patience:
                if kwargs.get("verbose", True):
                    print(f"Early stopping at epoch {epoch+1}")
                self.model.load_state_dict(best_model_weights)
                break
                
        if "valid" in kwargs:
            return history_train, history_valid
        else:
            return history_train
                
    def _predict(self, x: list[str], **kwargs) -> torch.Tensor:
        if "embeddings" in self.params["network"]:
            x = self.preprocessor.get_indices(x)
            x = torch.tensor(x, dtype=torch.long, device=self.device)
        elif "glove" in self.params["network"]:
            x, mask = self.preprocessor(x)
        else:
            x = self.preprocessor(x).todense()
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        if "tnn" in self.params["network"]:
            y_pred = self.model(x, mask)
        else:
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
            disable_tqdm = not kwargs.get("verbose", True)
            batch_range = tqdm(range(0, len(x), batch_size), "Predicting", disable=disable_tqdm)
            for i in batch_range:
                start_index = i
                end_index = min(i + batch_size, len(x))
                x_batch = x[start_index:end_index]
                y_pred_batch = self._predict(x_batch, **kwargs)
                if "y" in kwargs:
                    y_true_batch = kwargs["y"][start_index:end_index]
                    y_true_batch = [self.classes_.index(label) for label in y_true_batch]
                    y_true_batch = torch.tensor(y_true_batch, device=self.device)
                    loss = nn.CrossEntropyLoss()(y_pred_batch, y_true_batch)
                    batch_range.set_postfix(loss=loss.item())
                y_pred.append(y_pred_batch)
            y_pred = torch.cat(y_pred)
            labels = torch.argmax(y_pred, dim=1).cpu()
            str_labels = [self.classes_[label] for label in labels]
            return str_labels[0] if single_mode else str_labels
        
    def reset(self):
        self.__init__(**self.params)
        
    def save(self, path: str):
        with open(path, 'wb') as f:
            torch.save((self.params, self.model.state_dict(), self.preprocessor), f)
    
    def load(path: str) -> 'NeuralNetwork':
        with open(path, 'rb') as f:
            params, state_dict, preprocessor = torch.load(f)
        self = NeuralNetwork(**params)
        self.model.load_state_dict(state_dict)
        self.preprocessor = preprocessor
        return self
    
    def classes(self) -> list[str]:
        return self.classes_
    