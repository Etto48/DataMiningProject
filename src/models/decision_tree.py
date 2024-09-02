from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from dmml_project.dataset import Dataset
from dmml_project.models import Hyperparameters, Model
from dmml_project.preprocessor import Preprocessor

class DecisionTree(Model):
    def __init__(self, params: Hyperparameters):
        self.tree = DecisionTreeClassifier(
            criterion=params["decision_tree"]["criterion"],
            splitter=params["decision_tree"]["splitter"])
        self.preprocessor = Preprocessor.load(f"data/preprocessor/{params["decision_tree"]["preprocessor"]}.pkl")
    
    def train(self, train: Dataset, **kwargs):
        self.tree.set_params(class_weight=train.class_weights())
        self.tree.set_params(**kwargs)
        self.tree.fit(self.preprocessor(train.get_x()), train.get_y())
    
    def evaluate(self, valid: Dataset, **kwargs) -> float:
        return self.tree.score(self.preprocessor(valid.get_x()), valid.get_y())
    
    def predict(self, x: str, **kwargs) -> str:
        self.tree.predict(self.preprocessor([x]))
    def save(self, path: str):
        raise NotImplementedError
    def load(self, path: str):
        raise NotImplementedError
    
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    dataset = Dataset.load("data/crowdflower.tsv")
    dt = DecisionTree(Hyperparameters({"decision_tree": {"criterion": "gini", "splitter": "best", "preprocessor": "binary"}}, "tree"))
    xval = dt.xval(dataset, 10)
    mean = np.mean(xval)
    std = np.std(xval)
    plt.plot(xval)
    plt.axhline(mean, color="green")
    plt.fill_between(range(len(xval)), mean - std, mean + std, color="orange", alpha=0.5)
    plt.show()
    
    