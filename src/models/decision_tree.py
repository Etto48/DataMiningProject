from typing import Any, Callable, Literal
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from dmml_project.dataset import Dataset
from dmml_project.models import Model
from dmml_project.preprocessor import Preprocessor
from dmml_project import PROJECT_ROOT
import numpy as np

class DecisionTree(Model):
    def __init__(self, **kwargs):
        self.params = kwargs
        self.tree = DecisionTreeClassifier(**kwargs)
        self.tree.set_params(class_weight="balanced")
        self.preprocessor = Preprocessor.load(f"{PROJECT_ROOT}/data/preprocessor/binary.pkl")
    
    def train(self, train: Dataset, **kwargs):
        self.tree.fit(self.preprocessor(train.get_x()), train.get_y())
    def predict(self, x: list[str] | str, **kwargs) -> np.ndarray | int:
        if isinstance(x, str):
            x = [x]
        ret = self.tree.predict(self.preprocessor(x))
        return ret[0] if isinstance(x, str) else ret
    def reset(self):
        self.__init__(**self.params)
    def save(self, path: str):
        raise NotImplementedError
    def load(self, path: str):
        raise NotImplementedError
    def classes(self) -> list[str]:
        return self.tree.classes_

    