HYPERPARAMETERS = [{ # Set 0
    "decision_tree": {
        "criterion": ["gini", "entropy", "log_loss"],
        "splitter": ["best"],
        "max_depth": [10, 100, None],
        "min_impurity_decrease": [1e-2, 1e-4, 1e-6, 1e-8, 0],
        "class_weight": ["balanced"]
    },
    "random_forest": {
        "n_estimators": [10, 50, 100],
        "criterion": ["gini", "entropy", "log_loss"],
        "max_depth": [10, 100, None],
        "min_impurity_decrease": [1e-2, 1e-4, 1e-6, 0],
        "n_jobs": [-1],
        "class_weight": ["balanced"],
    },
    "neural_network": {
        "network": ["ff_tfidf", "lstm_embeddings", "lstm_glove"],
        "base_size": [8, 16, 32],
        "depth": [1, 2, 3, 4],
        "epochs": [8, 10, 15],
        "dropout": [0.25, 0.5],
        "batchnorm": [True, False],
        "batch_size": [32],
        "lr": [1e-2, 1e-3],
        "optimizer": ["adam", "sgd"],
    }
},
{ # Set 1
    "decision_tree": {
        "criterion": ["gini", "log_loss"],
        "splitter": ["best"],
        "max_depth": [1000, None],
        "min_impurity_decrease": [1e-5, 1e-6, 1e-7],
        "class_weight": ["balanced"]
    },
    "random_forest": {
        "n_estimators": [100, 125, 150],
        "criterion": ["gini", "log_loss"],
        "max_depth": [1000, None],
        "min_impurity_decrease": [1e-5, 1e-6, 1e-7],
        "n_jobs": [-1],
        "class_weight": ["balanced"],
    },
    "neural_network": {
        "network": ["ff_tfidf", "lstm_embeddings", "lstm_glove"],
        "base_size": [8, 16],
        "depth": [2, 3, 4],
        "epochs": [10, 15],
        "dropout": [0.25, 0.5],
        "batchnorm": [True, False],
        "batch_size": [32],
        "lr": [1e-2, 1e-3],
        "optimizer": ["adam"],
    }
}]

if __name__ == "__main__":
    import sys
    from itertools import product
    total_configs = 0
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    for model, params in HYPERPARAMETERS[n].items():
        print(f"Model: {model}")
        for key, values in params.items():
            print(f"\t{key}: {values}")
        configs = len(list(product(*params.values())))
        print()
        print(f"\tTotal configurations for {model}: {configs}")
        print()
        total_configs += configs
    print(f"Total configurations across all models: {total_configs}")