import pandas as pd

class Dataset:
    def __init__(self, path: str):
        self.data = pd.read_csv(path, sep="\t", encoding="ISO-8859-1")
    
    def get_x(self) -> list[str]:
        return self.data["text"].tolist()
    
    def get_y(self) -> list[str]:
        return self.data["label"].tolist()
    
    def __getitem__(self, key) -> tuple[str, str]:
        return self.data.iloc[key]
    
    def __len__(self) -> int:
        return len(self.data)