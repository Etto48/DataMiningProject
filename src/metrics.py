import sklearn.metrics

def f1_score(y_true, y_pred):
    """
    Calculate the F1 score.

    ## Parameters:
    - `y_true` (`Any`): The true labels.
    - `y_pred` (`Any`): The predicted labels.

    ## Returns:
    - `Any`: The F1 score.
    """
    return sklearn.metrics.f1_score(y_true, y_pred, average="weighted")