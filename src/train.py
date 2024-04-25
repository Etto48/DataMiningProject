from dmml.dataset import Dataset
from dmml.models.model import Model


def train_model(model: Model, dataset: Dataset):
    model.fit(dataset)