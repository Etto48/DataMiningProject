import json
import numpy as np
from dmml_project.model_selection.load_results import load_results, get_results_indices, model_name_from_index, index_from_model_name
import argparse
import regex as re

def print_config_results(search_results, index):
    result = search_results[index[0]][index[1]][index[2]]
    model_name = model_name_from_index(index)
    print(f"Model: {model_name}")
    print()
    hypers = result[0]
    accuracies = result[1]
    accuracy_avg = np.mean(accuracies)
    min_accuracy = np.min(accuracies)
    max_accuracy = np.max(accuracies)
    print(f"Hyperparameters: {json.dumps(hypers, indent=4)}")
    print()
    print(f"Accuracy: {accuracy_avg:.2f} [{min_accuracy:.2f}, {max_accuracy:.2f}]")

def names_match(model_name, query) -> bool:
    query_regex = re.escape(query).replace("\\*", ".*")
    return re.match(query_regex, model_name) is not None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print the results of the model selection.")
    parser.add_argument("--query", "-q", type=str, help="Query the results for a specific model.")
    parser.add_argument("--pause", "-p", action="store_true", help="Pause between each model and wait for the Enter key.")
    args = parser.parse_args()
    
    search_results = load_results(verbose=False)
    indices = get_results_indices(search_results)
    
    if args.query is None:
        for index in indices:
            print_config_results(search_results, index)
            print()
            if args.pause:
                input("Press Enter to continue...")
            print()
    elif "*" in args.query:
        num_matches = 0
        for index in indices:
            model_name = model_name_from_index(index)
            if names_match(model_name, args.query):
                num_matches += 1
                print_config_results(search_results, index)
                print()
                if args.pause:
                    input("Press Enter to continue...")
                print()
        if num_matches == 0:
            print("No matches found.")
    else:
        index = index_from_model_name(args.query, search_results)
        print_config_results(search_results, index)