from __future__ import annotations
import os
import random
import toml

# Genetic hyperparameter optimization
# @INPROCEEDINGS{9504761,
#   author={Alibrahim, Hussain and Ludwig, Simone A.},
#   booktitle={2021 IEEE Congress on Evolutionary Computation (CEC)}, 
#   title={Hyperparameter Optimization: Comparing Genetic Algorithm against Grid Search and Bayesian Optimization}, 
#   year={2021},
#   volume={},
#   number={},
#   pages={1551-1559},
#   keywords={Training;Machine learning algorithms;Neural networks;Prediction algorithms;Search problems;Time measurement;Bayes methods;Hyperparmeter optimization;Grid Search;Bayesian;Genetic Algorithm},
#   doi={10.1109/CEC45853.2021.9504761}}


class HyperparameterFactory:
    def __init__(self, path: str = "hyperparameters_config.toml"):
        self.path = path
        self.settings = toml.load(self.path)
        self.last_population = None
        self.last_scores = None
        self.gen = 0
        
    def _get_max_value_from_dict(spec_dict: dict, key: str) -> any:
        value_type = spec_dict.get(f"{key}_type", "categorical")
        value = spec_dict[key]
        if isinstance(value, list):
            match value_type:
                case "categorical":
                    return value[-1]
                case "discrete":
                    return value[1]
                case "continuous":
                    return value[1]
                case _:
                    raise ValueError("Unknown value type")
        else:
            raise ValueError("Value must be a list")
        
    def _get_max_dict(spec_dict: dict) -> dict:
        output_dict = {}
        for (key, value) in spec_dict.items():
            if isinstance(value, list):
                output_dict[key] = HyperparameterFactory._get_max_value_from_dict(spec_dict, key)
            elif isinstance(value, dict):
                output_dict[key] = HyperparameterFactory._get_max_dict(value)
            else:
                output_dict[key] = value
        return output_dict
        
    def get_max_config(self) -> Hyperparameters:
        return Hyperparameters(HyperparameterFactory._get_max_dict(self.settings), "max")
    
    def _get_random_value_from_dict(input_dict: dict, key: str):
        type_key = f"{key}_type"
        type_value = input_dict.get(type_key, "categorical")
        
        value = input_dict[key]
        assert isinstance(value, list)
        
        match type_value:
            case "categorical":
                return random.choice(value)
            case "discrete":
                if len(value) == 2:
                    return random.randint(value[0], value[1])
                elif len(value) == 3:
                    return random.choice(range(value[0], value[1] + 1, value[2]))
                else:
                    raise ValueError("Discrete values must have 2 (min,max) or 3 (min, max, step) elements")
            case "continuous":
                if len(value) == 2:
                    return random.uniform(value[0], value[1])
                else:
                    raise ValueError("Continuous values must have 2 (min,max) elements")
            case _:
                raise ValueError("Unknown value type")
        
    def _get_random_dict(input_dict: dict) -> dict:
        output_dict = {}
        for (key, value) in input_dict.items():
            if isinstance(value, list):
                output_dict[key] = HyperparameterFactory._get_random_value_from_dict(input_dict, key)
            elif isinstance(value, dict):
                output_dict[key] = HyperparameterFactory._get_random_dict(value)
            else:
                output_dict[key] = value
        return output_dict
                
    def _get_mutated_dict(spec_dict: dict, mutable_dict: dict, mutation_rate: float) -> dict:
        output_dict = {}
        for (key, value) in spec_dict.items():
            if isinstance(value, list):
                is_mutated = random.random() < mutation_rate
                if is_mutated:
                    output_dict[key] = HyperparameterFactory._get_random_value_from_dict(spec_dict, key)
                else:
                    output_dict[key] = mutable_dict[key]
            elif isinstance(value, dict):
                output_dict[key] = HyperparameterFactory._get_mutated_dict(value, mutable_dict[key], mutation_rate)
            else:
                output_dict[key] = value
        return output_dict
        
    def _get_identifier(gen: int, i: int) -> str:
        return f"gen{gen}_{i}"
    
    def _get_chromosome(individual_dict: dict) -> list[str]:
        toml_string = toml.dumps(individual_dict)
        return toml_string.split("\n")
    
    def _get_dict_from_chromosome(chromosome: list[str]) -> dict:
        toml_string = "\n".join(chromosome)
        return toml.loads(toml_string)
                
    def _mix_chromosomes(chromosome1: list[str], chromosome2: list[str], crossover_rate: float) -> list[str]:
        if len(chromosome1) != len(chromosome2):
            raise ValueError("Chromosomes must be of the same length")
        mixed_chromosome = []
        read_from_first = random.choice([True, False])
        for i in range(len(chromosome1)):
            # keys must match, if they don't contain an = this should also pass as long as they are the same string
            if chromosome1[i].split("=")[0] != chromosome2[i].split("=")[0]:
                raise ValueError("Chromosomes must have the same keys")

            if random.random() < crossover_rate:
                read_from_first = not read_from_first
            mixed_chromosome.append(chromosome1[i] if read_from_first else chromosome2[i])
        return mixed_chromosome
    
    def _mutate_chromosome(spec_dict: dict, chromosome: list[str], mutation_rate: float) -> list[str]:
        chromosome_dict = HyperparameterFactory._get_dict_from_chromosome(chromosome)
        mutated_dict = HyperparameterFactory._get_mutated_dict(spec_dict, chromosome_dict, mutation_rate)
        new_chromosome = HyperparameterFactory._get_chromosome(mutated_dict)
        return new_chromosome

    def _mix_and_mutate_chromosomes(self, chromosome1: list[str], chromosome2: list[str], crossover_rate: float, mutation_rate: float) -> list[str]:
        mixed_chromosome = HyperparameterFactory._mix_chromosomes(chromosome1, chromosome2, crossover_rate)
        mutated_chromosome = HyperparameterFactory._mutate_chromosome(self.settings, mixed_chromosome, mutation_rate)
        return mutated_chromosome
        
        
    def generate_random_individual(self, id: str) -> Hyperparameters:
        return Hyperparameters(
            HyperparameterFactory._get_random_dict(self.settings),
            id
        )
        
    def _calculate_selection_probability(self, scores: list[float]) -> list[float]:
        # normalize scores to 0-1
        min_score = min(scores)
        max_score = max(scores)
        norm_scores = [(score - min_score) / (max_score - min_score) for score in scores]
        
        # we want to select with higher probability for lower scores so weight = 1-norm_score
        norm_scores = [1 - score for score in norm_scores]
        # calculate percentage
        total_score = sum(norm_scores)
        return [score / total_score for score in norm_scores]
    
    def _generate_child_individual(self, individual1: Hyperparameters, individual2: Hyperparameters, i: int) -> Hyperparameters:
        chromosome1 = HyperparameterFactory._get_chromosome(individual1.settings)
        chromosome2 = HyperparameterFactory._get_chromosome(individual2.settings)
        output_chromosome = self._mix_and_mutate_chromosomes(chromosome1, chromosome2, self.settings["hyperparameter_search"]["genetic"]["crossover_rate"], self.settings["hyperparameter_search"]["genetic"]["mutation_rate"])
        output_dict = HyperparameterFactory._get_dict_from_chromosome(output_chromosome)
        return Hyperparameters(
            output_dict,
            HyperparameterFactory._get_identifier(self.gen, i)
        )
        
    def load_generation(self, gen: int) -> list[Hyperparameters]:
        self.gen = gen
        population_size = self.settings["hyperparameter_search"]["genetic"]["population_size"]
        self.last_population = [Hyperparameters(toml.load(f"history/gen{gen}_{i}/hyperparameters.toml"), f"gen{gen}_{i}") for i in range(population_size)]
        self.last_scores = None
        return self.last_population
        
    def next_generation(self) -> list[Hyperparameters]:
        population_size = self.settings["hyperparameter_search"]["genetic"]["population_size"]
        if self.last_population is None:
            self.last_population = [self.generate_random_individual(HyperparameterFactory._get_identifier(self.gen, i)) for i in range(population_size)]
        elif self.last_scores is None:
            raise ValueError("Scores not set, use set_scores to set scores first")
        elif len(self.last_scores) != population_size or len(self.last_population) != population_size:
            raise ValueError("Scores and population size do not match")
        else:
            new_population = []
            probabilities = self._calculate_selection_probability(self.last_scores)
            for i in range(population_size):
                [parent1, parent2] = random.choices(self.last_population, weights=probabilities, k=2)
                new_individual = self._generate_child_individual(parent1, parent2, i)
                new_population.append(new_individual)
            self.last_population = new_population
            
        self.gen += 1
        self.last_scores = None
        return self.last_population
    
    def set_scores(self, scores: list[float]):
        if self.last_scores is None:
            self.last_scores = scores
        else:
            raise ValueError("Scores already set, use next_generation to generate new population first")
        
    def get_best(self) -> Hyperparameters:
        if self.last_scores is None:
            raise ValueError("Scores not set, use set_scores to set scores first")
        else:
            return self.last_population[self.last_scores.index(min(self.last_scores))]
        
    def __getitem__(self, name: str) -> any:
        return Hyperparameters(self.settings, "__HYPERPARAMETER_FACTORY__")[name]
        

class Hyperparameters:
    def from_config(path: str = "test/hyperparameters.toml") -> Hyperparameters:
        file_name = os.path.basename(path).split(".")[0]
        self = Hyperparameters({}, file_name)
        self.path = path
        self.settings = toml.load(self.path)
        return self
    
    def __init__(self, settings: dict, identifier: str):
        self.path = None
        if settings is None or identifier is None:
            raise ValueError("settings and identifier must be set")
        self.settings = settings
        self.identifier = identifier
        
    def __getitem__(self, name):
        if name not in self.settings:
            match name:
                case "identifier":
                    return self.identifier
                case _:
                    raise ValueError(f"Unknown hyperparameter {name}")
        else:    
            return self.settings[name]
            
    def __repr__(self):
        return f"Hyperparameters({self.settings})"
    
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok = True)
        
        with open(path, "w") as f:
            toml.dump(self.settings, f)