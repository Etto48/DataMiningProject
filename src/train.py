from dmml.dataset import Dataset
from dmml.models.hyperparameters import HyperparameterFactory, Hyperparameters
from dmml.models.model import Model

def hyperparameter_search(train: Dataset, valid: Dataset) -> Hyperparameters:
    hf = HyperparameterFactory()
    iterations = hf["hyperparameter_search"]["iterations"]
    best_params = None
    best_accuracy = 0
    
    for i in range(iterations):
        params = hf.generate_random_individual(f"rnd_{i}")
        model = Model(params)
        
        model.train(train)
        accuracy = model.evaluate(valid)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params
        
    return best_params
        
        