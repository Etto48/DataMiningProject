from dmml_project.models.create_model import create_model
from dmml_project.dataset import Dataset
from dmml_project import PROJECT_ROOT
from dmml_project.models.hyperparameters import HYPERPARAMETERS
from itertools import product
import numpy as np

SEARCH_ITERATIONS = 50
RANDOM_SEED = 42

if __name__ == "__main__":
    dataset: Dataset = Dataset.load(f"{PROJECT_ROOT}/data/crowdflower.tsv")
    output_file = f"{PROJECT_ROOT}/data/model_selection.json"
    np.random.seed(RANDOM_SEED)
    for model, hypers in HYPERPARAMETERS.items():
        possible_hyperparameters = list(product(*hypers.values()))
        np.random.shuffle(possible_hyperparameters)
        max_index = min(SEARCH_ITERATIONS, len(possible_hyperparameters))
        for i in range(max_index):
            hyperparameters = dict(zip(hypers.keys(), possible_hyperparameters[i]))
            model_object = create_model(model, **hyperparameters)
            fold_metrics = model_object.xval(dataset, 10)
            

            