import json
from dmml_project import PROJECT_ROOT
from dmml_project.models.hyperparameters import HYPERPARAMETERS

if __name__ == "__main__":
    for model_kind in HYPERPARAMETERS.keys():
        path = f"{PROJECT_ROOT}/data/{model_kind}_search.json"
        with open(path, 'r') as f:
            data = json.load(f)
            data["search_results"]
            
            