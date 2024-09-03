from dmml_project.models import DecisionTree, RandomForest, NeuralNetwork, Model

def create_model(kind:str, **kwargs) -> Model:
    if kind == "decision_tree":
        return DecisionTree(**kwargs)
    elif kind == "random_forest":
        return RandomForest(**kwargs)
    elif kind == "neural_network":
        return NeuralNetwork(**kwargs)
    else:
        raise ValueError(f"Unknown model kind '{kind}'")