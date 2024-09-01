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
        self.tree.fit(self.preprocessor(train.get_x()), train.get_y())
    
    def evaluate(self, valid: Dataset, **kwargs) -> float:
        return self.tree.score(self.preprocessor(valid.get_x()), valid.get_y())
    
    def predict(self, x: str, **kwargs) -> str:
        raise NotImplementedError
    def save(self, path: str):
        raise NotImplementedError
    def load(self, path: str):
        raise NotImplementedError