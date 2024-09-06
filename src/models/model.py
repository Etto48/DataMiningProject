from __future__ import annotations
from typing import Any, Callable
from sklearn.metrics import accuracy_score
from dmml_project.dataset import Dataset
from tqdm import tqdm
import numpy as np

class Model:
    def __init__(self, params: dict[str, Any], **default_args) -> None:
        """
        Initialize the model.
        
        ## Parameters:
        - `**kwargs`: Additional keyword arguments to pass to the model.
        """
        for key, value in default_args.items():
            if key not in params:
                params[key] = value
        
        self.params = params
    
    def get_params(self) -> dict:
        """
        Get the model's parameters.
        
        ## Returns:
        - `dict`: The model's parameters.
        """
        return self.params
        
    def _not_implemented(self, method: str) -> NotImplementedError:
        return NotImplementedError(f"Model [{type(self).__name__}] must implement the required {method} method")
    def train(self, train: Dataset, **kwargs):
        """
        Train the model on the dataset.
        
        ## Parameters:
        - `train` (`Dataset`): The dataset to train the model on.
        - `**kwargs`: Additional keyword arguments to pass to the model.
        """
        raise self._not_implemented("train")
    def evaluate(self, valid: Dataset, metric: Callable[[Any, Any], Any] = accuracy_score, **kwargs) -> Any:
        """
        Evaluate the model on the dataset.

        ## Parameters:
        - `valid` (`Dataset`): The dataset to evaluate the model on.
        - `metric` (`Callable[[Any, Any], Any]`): The metric to use for evaluation.
        - `**kwargs`: Additional keyword arguments to pass to the model.
        
        ## Returns:
        - `Any`: The return of the metric function.
        """
        true_y = valid.get_y()
        predicted_y = self.predict(valid.get_x())
        return metric(true_y, predicted_y)
    def predict(self, x: list[str] | str, **kwargs) -> np.ndarray | str:
        """
        Predict the labels of the input data.
        
        ## Parameters:
        - `x` (`list[str] | str`): The input data to predict.
        - `**kwargs`: Additional keyword arguments to pass to the model.
        
        ## Returns:
        - `np.ndarray | str`: The predicted labels, it's a string if the input is a single string, 
        otherwise it's a numpy array of strings.
        """
        raise self._not_implemented("predict")
    def reset(self):
        """
        Reset the model to its initial state.
        """
        raise self._not_implemented("reset")
    def save(self, path: str):
        """
        Save the model to the specified path.
        
        ## Parameters:
        - `path` (`str`): The path to save the model to.
        """
        raise self._not_implemented("save")
    def load(path: str) -> Model:
        """
        Load the model from the specified path.
        
        ## Parameters:
        - `path` (`str`): The path to load the model from.
        
        ## Returns:
        - `Model`: The loaded model.
        """
        raise NotImplementedError
    def classes(self) -> list[str]:
        raise self._not_implemented("classes")
    def xval(self, dataset: Dataset, folds: int = 10, metric: Callable[[Any, Any], Any] = accuracy_score, **kwargs) -> list[Any]:
        """
        Perform cross validation training and evaluation on the dataset.
        
        ## Parameters:
        - `dataset` (`Dataset`): The dataset to perform cross validation on.
        - `folds` (`int`): The number of folds to use in cross validation.
        - `metric` (`Callable[[Any, Any], Any]`): The metric to use for evaluation.
        - `**kwargs`: Additional keyword arguments to pass to the model.
        
        ## Returns:
        - `list[Any]`: The evaluation results for each fold, as returned by the metric function.
        """
        evaluations: list = []
        for i in tqdm(range(folds), desc="Cross validation"):
            self.reset()
            train, valid = dataset.fold(i, folds)
            self.train(train, valid=valid, verbose=False, **kwargs)
            evaluations.append(self.evaluate(valid, metric, verbose=False))
        self.reset()
        return evaluations
    