import rich.live
import rich.progress
from dmml.dataset import Dataset
from dmml.models import HyperparameterFactory, Hyperparameters, Model, create_model
import rich

def train_fold(model: Model, train: Dataset, valid: Dataset) -> float:
    model.train(train)
    accuracy = model.evaluate(valid)
    return accuracy

def hyperparameter_search(progress: rich.progress.Progress, data: Dataset) -> Hyperparameters:
    hf = HyperparameterFactory()
    iterations = hf["hyperparameter_search"]["iterations"]
    best_params = None
    best_accuracy = 0
    search_task = progress.add_task("Search", total=iterations)
    for i in range(iterations):
        params = hf.generate_random_individual(f"rnd_{i}")
        accuracy = 0
        
        fold_task = progress.add_task(f"Fold", total=hf["folds"])
        for i in range(hf["folds"]):
            model = create_model(params)
            train, valid = data.fold(i, hf["folds"])
            train_fold(model, train, valid)
            accuracy += model.evaluate(valid)
            progress.update(fold_task, advance=1)
            
        progress.remove_task(fold_task)
        accuracy /= hf["folds"]
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params
            
        progress.update(search_task, advance=1)
    progress.remove_task(search_task)    
    
    return best_params
        
        
def main():
    progress = rich.progress.Progress(
        rich.progress.TextColumn("[blue]{task.description}", justify="right"),
        rich.progress.MofNCompleteColumn(),
        rich.progress.BarColumn(),
        rich.progress.TaskProgressColumn() ,
        rich.progress.TimeRemainingColumn(),
        rich.progress.TimeElapsedColumn(),
        speed_estimate_period=60*60*24*10,
        refresh_per_second=2
    )
    live = rich.live.Live(progress, refresh_per_second=2)
    with live:
        best = hyperparameter_search(progress, Dataset.load("data/crowdflower.tsv"))
    best.save("hyperparameters/best.toml")


if __name__ == "__main__":
    main()