from complete import complete
import json
import os
import random


def mutate_or_keep_val(val: str | float, mutation_prompt: str, mutation_rate: float):
    if random.random() < mutation_rate:
        if isinstance(val, str):
            return complete(system_prompt=mutation_prompt, user_prompt=val)
        elif isinstance(val, float):
            result = complete(system_prompt=mutation_prompt, user_prompt=str(val))
            # if result is a number, return a float
            if result.isdigit():
                return float(result)
            else:
                return val
    else:
        return val


def mutate_genome():
    # load genome.json
    with open(os.path.join(os.path.dirname(__file__), "genome.json"), "r") as f:
        genome = json.load(f)

    mutation_rate = genome["parameters"]["mutation_rate"]
    mutation_prompt = genome["parameters"]["mutation_prompt"]

    new_genome = {}
    for key, value in genome.items():
        if isinstance(value, str):
            new_genome[key] = mutate_or_keep_val(value, mutation_prompt, mutation_rate)
        elif isinstance(value, list):
            new_genome[key] = [
                mutate_or_keep_val(item, mutation_prompt, mutation_rate)
                for item in value
            ]
        elif isinstance(value, dict):
            new_genome[key] = {
                k: mutate_or_keep_val(v, mutation_prompt, mutation_rate)
                for k, v in value.items()
            }
    return new_genome


if __name__ == "__main__":
    print(mutate_genome())
