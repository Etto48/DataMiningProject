HYPERPARAMETERS = {
    "decision_tree": {
        "criterion": ["gini", "entropy", "log_loss"],
        "splitter": ["best"],
        "max_depth": [10, 100, None],
        "min_impurity_decrease": [1e-2, 1e-4, 1e-6, 1e-8, 0],
        "class_weight": ["balanced"]
    },
    "random_forest": {
        "n_estimators": [10, 100, 1000],
        "criterion": ["gini", "entropy", "log_loss"],
        "max_depth": [10, 100, None],
        "min_impurity_decrease": [1e-2, 1e-4, 1e-6, 1e-8, 0],
        "n_jobs": [-1],
        "class_weight": ["balanced"],
    },
    "neural_network": {
        "network": ["ff_tfidf", "ff_binary", "lstm_embeddings", "cnn_embeddings"],
        "base_size": [8, 16, 32, 64],
        "depth": [2, 3, 4],
        "epochs": [5, 10, 15],
        "dropout": [0, 0.5],
        "batchnorm": [True, False],
        "batch_size": [32],
        "lr": [1e-1, 1e-2, 1e-3],
        "optimizer": ["adam", "sgd"],
    }
}

if __name__ == "__main__":
    from itertools import product
    total_configs = 0
    for model, params in HYPERPARAMETERS.items():
        print(f"Model: {model}")
        for key, values in params.items():
            print(f"\t{key}: {values}")
        configs = len(list(product(*params.values())))
        print()
        print(f"\tTotal configurations for {model}: {configs}")
        print()
        total_configs += configs
    print(f"Total configurations across all models: {total_configs}")