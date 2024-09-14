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
        "network": ["ff_tfidf", "lstm_embeddings"],
        "base_size": [8, 16, 32],
        "depth": [1, 2, 3, 4],
        "epochs": [10, 15],
        "dropout": [0.5],
        "batchnorm": [True, False],
        "batch_size": [32],
        "lr": [1e-2, 1e-3],
        "optimizer": ["adam"],
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
        "network": ["ff_tfidf", "lstm_glove"],
        "base_size": [16, 32],
        "depth": [1, 2, 3],
        "epochs": [20],
        "patience": [2],
        "dropout": [0.5],
        "batchnorm": [False],
        "batch_size": [32],
        "lr": [1e-3],
        "optimizer": ["adam"],
    }
}]

CAPTIONS = [
    {
        "decision_tree": "Decision Trees hyperparameter search space for the first search.",
        "random_forest": "Random Forest hyperparameter search space for the first search.",
        "neural_network": "Neural Networks hyperparameter search space for the first search. While using the \\texttt{ff\\_tfidf} network, if \\texttt{batchnorm} is set to \\texttt{True}, the \\texttt{dropout} hyperparameter is set to $0$. When using the \\texttt{lstm\\_embeddings} and \\texttt{lstm\\_glove} networks, the \\texttt{batchnorm} hyperparameter is set to \\texttt{False}."
    },
    {
        "decision_tree": "Decision Trees hyperparameter search space for the second search.",
        "random_forest": "Random Forest hyperparameter search space for the second search.",
        "neural_network": "Neural Networks hyperparameter search space for the second search. The same rules described in \\autoref{tab:hyperparameters_neural_network_0} apply."
    }
]

if __name__ == "__main__":
    from itertools import product
    from dmml_project.utils import hyper_range_to_latex
    from dmml_project import PAPER_TABLES
    
    with open(f"{PAPER_TABLES}/hypers.tex", "w") as f:
        f.write(hyper_range_to_latex(HYPERPARAMETERS, CAPTIONS))
    
    for n in range(len(HYPERPARAMETERS)):
        total_configs = 0
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