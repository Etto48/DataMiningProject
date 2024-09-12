from __future__ import annotations
from dmml_project import PROJECT_ROOT, CLASSES
import pandas as pd

class Dataset:
    def __init__(self):
        self.data = pd.DataFrame()
    
    def from_x_y(self, x: list[str], y: list[str]) -> Dataset:
        self = Dataset()
        self.data = pd.DataFrame({"text": x, "label": y})
        return self
    
    def load(path: list[str] | str) -> Dataset:
        self = Dataset()
        path = path if isinstance(path, list) else [path]
        self.data = pd.DataFrame(columns=["text", "label"])
        for p in path:
            if p.endswith(".tsv"):
                new_data = pd.read_csv(p, sep="\t", encoding="ISO-8859-1")
                self.data = pd.concat([self.data, new_data], ignore_index=True)
        
        # shuffle the data
        self.data = self.data.sample(frac=1, replace=False).reset_index(drop=True)
        
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
    
    def sfold(self, fold: int, total: int) -> tuple[Dataset, Dataset]:
        indices_per_class = dict([(k, []) for k in CLASSES])
        for i, row in self.data.iterrows():
            indices_per_class[row["label"]].append(i)
        
        train_indices = list()
        valid_indices = list()
        
        for k, v in indices_per_class.items():
            fold_size = len(v) // total
            start = fold * fold_size
            end = start + fold_size
        
            for i, idx in enumerate(v):
                if start <= i < end:
                    valid_indices.append(idx)
                else:
                    train_indices.append(idx)

        #train_idx_set = set(train_indices)
        #valid_idx_set = set(valid_indices)
        #assert len(train_idx_set.intersection(valid_idx_set)) == 0

        train = Dataset()
        train.data = self.data.iloc[train_indices]
        valid = Dataset()
        valid.data = self.data.iloc[valid_indices]
        
        return train, valid
    
    def batch(self, batch: int, batch_size: int) -> Dataset:
        start = batch * batch_size
        end = start + batch_size
        end = min(end, len(self.data))
        
        batch_data = self.data.iloc[start:end]
        batch_dataset = Dataset()
        batch_dataset.data = batch_data
        return batch_dataset
    
    def __getitem__(self, key) -> tuple[str, str]:
        return self.data.iloc[key]
    
    def __len__(self) -> int:
        return len(self.data)
    
    def class_distribution(self) -> dict[str, int]:
        return self.data["label"].value_counts().to_dict()
    
    def class_weights(self) -> dict[str, float]:
        dist = self.class_distribution()
        total = sum(dist.values())
        classes = len(dist)
        weights = {k: total / (classes * v) for k, v in dist.items()}
        return weights
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    dataset: Dataset = Dataset.load(f"{PROJECT_ROOT}/data/train.tsv")
    
    distribution = dataset.class_distribution()
    majority_class = max(distribution, key=distribution.get)
    total = len(dataset)
    
    print(f"Dataset size: {total}")
    print(f"Minimum accuracy to beat: {distribution[majority_class] / total * 100:.2f}%")
    
    plt.subplot(1, 2, 1)
    plt.title("Class distribution")
    plt.bar(dataset.class_distribution().keys(), dataset.class_distribution().values())
    plt.xticks(rotation=60, size=8)
    plt.subplot(1, 2, 2)
    plt.title("Class weights")
    plt.bar(dataset.class_weights().keys(), dataset.class_weights().values())
    plt.xticks(rotation=60, size=8)
    plt.show()