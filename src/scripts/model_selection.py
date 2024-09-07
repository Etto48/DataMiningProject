from dmml_project.dataset import Dataset
from dmml_project import PROJECT_ROOT
from dmml_project.models.hyperparameters import HYPERPARAMETERS
from dmml_project.model_selection.random_search import RandomSearch

NUM_GENERATIONS = 2

if __name__ == "__main__":
    dataset: Dataset = Dataset.load(f"{PROJECT_ROOT}/data/train.tsv")
    assert isinstance(HYPERPARAMETERS, list[dict]), "HYPERPARAMETERS must be a list of dictionaries"
    for gen, hypers in enumerate(HYPERPARAMETERS):
        for model_kind, hyper_grid in hypers.items():
            search = RandomSearch(model_kind, hyper_grid, n_iter=10)
            output_file = f"{PROJECT_ROOT}/data/{model_kind}_search_{gen}.json"
            result = search.search(dataset, output_path=output_file)
            if result == []:
                exit(1)
    

            