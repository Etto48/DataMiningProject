from dmml_project.dataset import Dataset
from dmml_project.models import Model
from sklearn.ensemble import RandomForestClassifier
from dmml_project.preprocessor import Preprocessor
from dmml_project import PROJECT_ROOT
import numpy as np
import pickle as pkl

class RandomForest(Model):
    def __init__(self, **kwargs):
        super().__init__(kwargs, class_weight="balanced", n_jobs=-1, preprocessor="binary")
        self.forest = RandomForestClassifier(**kwargs)
        self.preprocessor = Preprocessor(kind=self.params['preprocessor'])
        
    def train(self, train: Dataset, **kwargs):
        self.preprocessor.fit(train.get_x(), verbose=kwargs.get('verbose', True))
        self.forest.fit(self.preprocessor(train.get_x()), train.get_y())
        
    def predict(self, x: list[str] | str, **kwargs) -> np.ndarray | str:
        if isinstance(x, str):
            x = [x]
        ret = self.forest.predict(self.preprocessor(x))
        return ret[0] if isinstance(x, str) else ret
    
    def reset(self):
        self.__init__(**self.params)
        
    def save(self, path: str):
        with open(path, 'wb') as f:
            pkl.dump((self.params, self.forest, self.preprocessor), f)
            
    def load(path: str) -> 'RandomForest':
        with open(path, 'rb') as f:
            params, forest, preprocessor = pkl.load(f)
        self = RandomForest(**params)
        self.forest = forest
        self.preprocessor = preprocessor
        return self
    
    def classes(self) -> list[str]:
        return self.forest.classes_
    