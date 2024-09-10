from dmml_project.models.hyperparameters import HYPERPARAMETERS
from dmml_project import PROJECT_ROOT
import os

if __name__ == "__main__":
    files = []
    for gen, gen_dict in enumerate(HYPERPARAMETERS):
        for model in gen_dict.keys():
            files.append(f"{model}_search_{gen}.json")
    
    files = ",".join(files)

    os.system(f"scp \"ettore@EttoreG3:/home/ettore/Documents/Dev/DataMiningProject/data/{{{files}}}\" \"{PROJECT_ROOT}/data/\"")