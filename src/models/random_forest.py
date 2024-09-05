from dmml_project.dataset import Dataset
from dmml_project.models import Model
from sklearn.ensemble import RandomForestClassifier
from dmml_project.preprocessor import Preprocessor
from dmml_project import PROJECT_ROOT
import numpy as np
import pickle as pkl

class RandomForest(Model):
    def __init__(self, **kwargs):
        super().__init__(kwargs, class_weight="balanced", n_jobs=-1)
        self.forest = RandomForestClassifier(**kwargs)
        self.preprocessor = Preprocessor.load(f"{PROJECT_ROOT}/data/preprocessor/binary.pkl")
        
    def train(self, train: Dataset, **kwargs):
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
            pkl.dump((self.params, self.forest), f)
            
    def load(path: str) -> 'RandomForest':
        with open(path, 'rb') as f:
            params, forest = pkl.load(f)
        self = RandomForest(**params)
        self.forest = forest
        return self
    
    def classes(self) -> list[str]:
        return self.forest.classes_
    