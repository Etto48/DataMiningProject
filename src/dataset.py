from __future__ import annotations
from dmml_project import PROJECT_ROOT
import pandas as pd

class Dataset:
    def __init__(self):
        self.data = pd.DataFrame()
    
    def load(path: str) -> Dataset:
        self = Dataset()
        self.data = pd.read_csv(path, sep="\t", encoding="ISO-8859-1")
        self.data = self.data.sample(frac=1).reset_index(drop=True)
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
        unnormalized_weights = {k: total / v for k, v in dist.items()}
        total_weight = sum(unnormalized_weights.values())
        return {k: v / total_weight for k, v in unnormalized_weights.items()}
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    dataset: Dataset = Dataset.load(f"{PROJECT_ROOT}/data/crowdflower.tsv")
    
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