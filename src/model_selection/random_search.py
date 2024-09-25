import json
from typing import Any, Callable, Literal, Optional
from itertools import product

from dmml_project.metrics import f1_score
from dmml_project.dataset import Dataset
import numpy as np

from dmml_project.models.create_model import create_model

class RandomSearch:
    def __init__(self, model_kind: Literal["decision_tree", "random_forest", "neural_network"], param_grid: dict[str, list], n_iter: int = 10, folds: int = 6, metric: Callable[[Any, Any], Any] = f1_score, **kwargs):
        """
        Initialize the random search.

        ## Parameters:
        - `model_kind` (`str`): The kind of model to use.
        - `param_grid` (`dict[str, list]`): The parameter grid to search.
        - `n_iter` (`int`): The number of iterations to search.
        - `**kwargs`: Additional keyword arguments to pass to the model.
        """
        self.model_kind = model_kind
        self.param_grid = param_grid
        self.n_iter = n_iter
        self.folds = folds
        self.metric = metric
        self.kwargs = kwargs
        
    def search(self, train: Dataset, valid: Optional[Dataset] = None, output_path: Optional[str] = None) -> list[tuple[dict[str, Any], list[Any]]]:
        """
        Search for the best parameters.

        ## Parameters:
        - `train` (`Dataset`): This dataset will be used for k-fold cross-validation if `valid` is not provided.
        - `valid` (`Dataset`): This dataset will be used for validation if provided, this will override the number of folds set.
        - `output_path` (`str`): The path to log the results to, this file will be a json file with the results updated after each iteration.
        - `**kwargs`: Additional keyword arguments to pass to the model.
        
        ## Returns:
        - `list[tuple[dict[str, Any], list[Any]]]`: A list of tuples containing the parameters and their respective evaluation results.
        """
        
        try:
            with open(output_path, 'r') as f:
                backup = json.load(f)
                param_grid = backup["param_grid"]
                assert param_grid == self.param_grid, "The parameter grid in the backup file does not match the parameter grid set in the search"
                available_params = backup["available_params"]
                indices = backup["indices"]
                assert len(indices) == self.n_iter, "The number of iterations in the backup file does not match the number of iterations set in the search"
                search_results = backup["search_results"]
                assert len(search_results) <= self.n_iter, "The number of results in the backup file is greater than the number of iterations set in the search"
                starting_index = len(search_results)
        except FileNotFoundError:
            available_params = list(product(*self.param_grid.values()))
            indices = np.random.choice(len(available_params), size=self.n_iter, replace=False).tolist()
            search_results = []
            starting_index = 0
        except Exception as e:
            print(f"An error occurred while loading the backup file: {e}. \nPath: {output_path}")
            return []
        
        for iteration in range(starting_index, self.n_iter):
            print(f"Evaluating {self.model_kind}, with parameters {iteration + 1:>{len(str(self.n_iter))}}/{self.n_iter}")
            param_id = indices[iteration]
            params = {key: value for key, value in zip(self.param_grid.keys(), available_params[param_id])}
            model = create_model(self.model_kind, **params)
            
            if valid is not None:
                model.train(train)
                evaluation = [model.evaluate(valid, metric=self.metric)]
            else:
                evaluation = model.xval(train, folds=self.folds, metric=self.metric)
            
            search_results.append((params, evaluation))
            if output_path is not None:
                with open(output_path, 'w') as f:
                    json.dump({
                        "param_grid": self.param_grid,
                        "available_params": available_params,
                        "indices": indices,
                        "search_results": search_results,
                    }, f)
            
        return search_results