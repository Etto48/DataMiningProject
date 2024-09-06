from dmml_project.dataset import Dataset
from dmml_project import PROJECT_ROOT
from dmml_project.models.hyperparameters import HYPERPARAMETERS
from dmml_project.model_selection.random_search import RandomSearch
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform hyperparameter search for models")
    parser.add_argument("--hyper-set, -h", type=int, help="Index of hyperparameter set to use", default=0)
    args = parser.parse_args()
    
    dataset: Dataset = Dataset.load(f"{PROJECT_ROOT}/data/train.tsv")
    
    for model_kind, hyper_grid in HYPERPARAMETERS[args.hyper_set].items():
        search = RandomSearch(model_kind, hyper_grid, n_iter=10)
        postfix = "" if args.hyper_set == 0 else f"_{args.hyper_set}"
        output_file = f"{PROJECT_ROOT}/data/{model_kind}_search{postfix}.json"
        result = search.search(dataset, output_path=output_file)
        if result == []:
            exit(1)
    

            