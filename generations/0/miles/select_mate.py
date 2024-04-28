from complete import complete
import json
import os


def format_mates(mates: dict[str, str]):
    return "\n".join([f"{name}: {phenome}" for name, phenome in mates.items()])


# mates is a map of mate names to their phenotype expressions, returns the name of the selected mate
def select_phenome(mates: dict[str, str]) -> str:
    # load genome.json
    with open(os.path.join(os.path.dirname(__file__), "genome.json"), "r") as f:
        genome = json.load(f)

    prompt = "\n".join(genome["prompt_genes"])
    prompt += f"\n{genome['pick_mate_gene']}"
    result = complete(system_prompt=prompt, user_prompt=format_mates(mates))
    try:
        return json.loads(result)["selection"]
    except IndexError:
        return ""


import sys

if __name__ == "__main__":
    mates_input = sys.stdin.read()
    mates_dict = json.loads(mates_input)
    print(select_phenome(mates_dict))
