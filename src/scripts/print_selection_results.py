import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dmml_project import PROJECT_ROOT
from dmml_project.models.hyperparameters import HYPERPARAMETERS
from dmml_project.model_selection.load_results import load_results, get_results_indices, model_name_from_index, index_from_model_name
import argparse

def print_config_results(search_results, index):
    result = search_results[index[0]][index[1]][index[2]]
    model_name = model_name_from_index(index)
    print(f"Model: {model_name}")
    print()
    hypers = result[0]
    accuracies = result[1]
    accuracy_avg = np.mean(accuracies)
    accuracy_std = np.std(accuracies)
    print(f"Hyperparameters: {json.dumps(hypers, indent=4)}")
    print()
    print(f"Accuracy: {accuracy_avg:.2f} ± {accuracy_std:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print the results of the model selection.")
    parser.add_argument("--query", "-q", type=str, help="Query the results for a specific model.")
    parser.add_argument("--yes", "-y", action="store_true", help="Continue printing the results without pressing Enter.")
    args = parser.parse_args()
    
    search_results = load_results(verbose=False)
    indices = get_results_indices(search_results)
    
    if args.query is None:
        for index in indices:
            print_config_results(search_results, index)
            print()
            if not args.yes:
                input("Press Enter to continue...")
            print()
    else:
        index = index_from_model_name(args.query, search_results)
        print_config_results(search_results, index)