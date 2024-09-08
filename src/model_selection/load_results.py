from typing import Optional
from dmml_project.models.hyperparameters import HYPERPARAMETERS
from dmml_project import PROJECT_ROOT

import json

BEAUTIFUL_NAMES = dict([(model_kind, model_kind.replace("_", " ").title()) for model_kind in HYPERPARAMETERS[0].keys()])

def get_results_indices(results: list[dict[str, list[tuple[dict, list[float]]]]]) -> list[tuple[int, str, int]]:
    """
    Get the indices of the results for each model kind and generation.
    
    ## Parameters:
    - `results`: See the return value of `load_results`.
    
    ## Returns:
    - A list of tuples, where each tuple contains the generation, model kind, and model index for each result.
    
    """
    indices = []

    for gen, gen_results in enumerate(results):
        for model_kind, kind_results in gen_results.items():
            for i, _ in enumerate(kind_results):
                indices.append((gen, model_kind, i))
    
    return indices

def model_name_from_index(index: tuple[int, str, int]) -> str:
    """
    Get the model name from the index tuple.
    
    ## Parameters:
    - `index`: A tuple containing the generation, model kind, and model index.
    
    ## Returns:
    - A unique string representing the model.
    
    """
    return f"{index[1]}-G{index[0]}-{index[2]}"

def index_from_model_name(model_name: str, results: Optional[list[dict[str, list[tuple[dict, list[float]]]]]] = None) -> tuple[int, str, int]:
    """
    Get the index tuple from the model name.
    
    ## Parameters:
    - `model_name`: A unique string representing the model.
    - `results`: See the return value of `load_results`.
    
    ## Returns:
    - A tuple containing the generation, model kind, and model index.
    """
    
    index = model_name.split("-")
    if len(index) != 3:
        raise ValueError("The query must be in the format '<model_kind>-G<generation>-<model_index>'.")
    model_kind = index[0]
    assert index[1].startswith("G"), f"Generation must start with 'G'."
    gen = int(index[1][1:])
    model_index = int(index[2])
    if results is not None:    
        assert len(results) > gen, f"Generation {gen} not found."
        assert model_kind in results[gen], f"Model kind '{model_kind}' not found in generation {gen}."
        assert len(results[gen][model_kind]) > model_index, f"Model index {model_index} not found for model kind '{model_kind}' in generation {gen}."
    return gen, model_kind, model_index

def load_results(verbose: bool = True) -> list[dict[str, list[tuple[dict, list[float]]]]]:
    """
    Load the results of the hyperparameter search for each model kind and generation.
    
    ## Parameters:
    - `verbose`: Whether to print a message when a search result is not found.
    
    ## Returns:
    - A list of dictionaries, where each dictionary contains the search results for each model kind. 
    The values are lists of tuples, where each tuple contains the hyperparameters and the corresponding 
    validation scores. The return should be indexed like this: 
    `ret[generation][model_kind][model_index][0]` for hyperparameters and 1 for scores.
    
    """
    ret = []
    generations = len(HYPERPARAMETERS)
    model_kinds = HYPERPARAMETERS[0].keys()
    for gen in range(generations):
        search_results = {}
        for model_kind in model_kinds:
            path = f"{PROJECT_ROOT}/data/{model_kind}_search_{gen}.json"
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    search_results[model_kind] = data["search_results"]
            except FileNotFoundError:
                if verbose:
                    print(f"Skipping {model_kind} as no search results were found")
        ret.append(search_results)
    return ret

    