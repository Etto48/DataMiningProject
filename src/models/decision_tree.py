from sklearn.tree import DecisionTreeClassifier
from dmml_project.dataset import Dataset
from dmml_project.models import Model
from dmml_project.preprocessor import Preprocessor
from dmml_project import PROJECT_ROOT
import numpy as np
import pickle as pkl

class DecisionTree(Model):
    def __init__(self, **kwargs):
        super().__init__(kwargs, class_weight="balanced", preprocessor="binary")
        self.preprocessor = Preprocessor(kind=self.params['preprocessor'])
        kwargs.pop('preprocessor', None)
        self.tree = DecisionTreeClassifier(**kwargs)
    
    def train(self, train: Dataset, **kwargs):
        self.preprocessor.fit(train.get_x(), verbose=kwargs.get('verbose', True))
        self.tree.fit(self.preprocessor(train.get_x()), train.get_y())
        
    def predict(self, x: list[str] | str, **kwargs) -> np.ndarray | str:
        if isinstance(x, str):
            x = [x]
        ret = self.tree.predict(self.preprocessor(x))
        return ret[0] if isinstance(x, str) else ret
    
    def reset(self):
        self.__init__(**self.params)
        
    def save(self, path: str):
        with open(path, 'wb') as f:
            pkl.dump((self.params, self.tree, self.preprocessor), f)
            
    def load(path: str) -> 'DecisionTree':
        with open(path, 'rb') as f:
            params, tree, preprocessor = pkl.load(f)
        self = DecisionTree(**params)
        self.tree = tree
        self.preprocessor = preprocessor
        return self
    
    def classes(self) -> list[str]:
        return self.tree.classes_
    