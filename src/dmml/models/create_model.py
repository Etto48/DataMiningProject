from dmml.models import Hyperparameters, DecisionTree, RandomForest, NeuralNetwork, Model

def create_model(params: Hyperparameters) -> Model:
    if params["model"]["kind"] == "decision_tree":
        return DecisionTree(params)
    elif params["model"]["kind"] == "random_forest":
        return RandomForest(params)
    elif params["model"]["kind"] == "neural_network":
        return NeuralNetwork(params)
    else:
        raise ValueError(f"Unknown model kind '{params['kind']}'")