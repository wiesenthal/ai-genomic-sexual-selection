from complete import complete
import json
import os


def express_phenome():
    # load genome.json
    with open(os.path.join(os.path.dirname(__file__), "genome.json"), "r") as f:
        genome = json.load(f)

    prompt = "\n".join(genome["prompt_genes"])
    prompt += f"\n{genome['expression_gene']}"
    return complete(prompt)


if __name__ == "__main__":
    print(express_phenome())
