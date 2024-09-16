from dmml_project import PROJECT_ROOT, CLASSES
from dmml_project.models.decision_tree import DecisionTree
from dmml_project.models.random_forest import RandomForest
from dmml_project.models.neural_network import NeuralNetwork
from dmml_project.models.model import Model

if __name__ == "__main__":
    MODEL_NAME = "random_forest-G1-8"
    match MODEL_NAME.split("-")[0]:
        case "decision_tree":
            model_class = DecisionTree
        case "random_forest":
            model_class = RandomForest
        case "neural_network":
            model_class = NeuralNetwork
        case _:
            raise ValueError(f"Model [{MODEL_NAME}] is not supported")

    model: Model = model_class.load(f"{PROJECT_ROOT}/data/{MODEL_NAME}.pkl")
    try:
        while True:
            text = input("Enter a sentence to classify: ")
            label = model.predict(text)
            print(f"{label[0]}")   
    except KeyboardInterrupt:
        print("\nGoodbye!")