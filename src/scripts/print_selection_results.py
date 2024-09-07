import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dmml_project import PROJECT_ROOT
from dmml_project.models.hyperparameters import HYPERPARAMETERS

if __name__ == "__main__":
    search_results = {}
    beautiful_names = {
        "decision_tree": "Decision Tree",
        "random_forest": "Random Forest",
        "neural_network": "Neural Network",
    }
    for model_kind in beautiful_names.keys():
        path = f"{PROJECT_ROOT}/data/{model_kind}_search.json"
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                search_results[model_kind] = data["search_results"]
        except FileNotFoundError:
            print(f"Skipping {model_kind} as no search results were found")
    
    for model_kind, search_result in search_results.items():
        print(f"Model: {beautiful_names[model_kind]}")
        print()
        for i, result in enumerate(search_result):
            print(f"\tResult {i+1}/{len(search_result)}")
            hypers = result[0]
            accuracies = result[1]
            accuracy_avg = np.mean(accuracies)
            accuracy_std = np.std(accuracies)
            print(f"\tHyperparameters: \n{json.dumps(hypers, indent=4)}")
            print()
            print(f"\tAccuracy: {accuracy_avg:.2f} ± {accuracy_std:.2f}")
            print()
            input("Press Enter to continue...")
            print()
        print()
        