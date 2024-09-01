from __future__ import annotations
import pandas as pd

class Dataset:
    def __init__(self):
        self.data = pd.DataFrame()
    
    def load(path: str) -> Dataset:
        self = Dataset()
        self.data = pd.read_csv(path, sep="\t", encoding="ISO-8859-1")
        self.data.sample(frac=1).reset_index(drop=True)
        return self
    
    def get_x(self) -> list[str]:
        return self.data["text"].tolist()
    
    def get_y(self) -> list[str]:
        return self.data["label"].tolist()
    
    def fold(self, fold: int, total: int) -> tuple[Dataset, Dataset]:
        fold_size = len(self.data) // total
        train_data = list()
        valid_data = list()
        start = fold * fold_size
        end = start + fold_size
        
        for i in range(len(self.data)):
            if start <= i < end:
                valid_data.append(self.data.iloc[i])
            else:
                train_data.append(self.data.iloc[i])

        train = Dataset()
        train.data = pd.DataFrame(train_data, columns=self.data.columns)
        valid = Dataset()
        valid.data = pd.DataFrame(valid_data, columns=self.data.columns)
    
        return train, valid
    
    def __getitem__(self, key) -> tuple[str, str]:
        return self.data.iloc[key]
    
    def __len__(self) -> int:
        return len(self.data)