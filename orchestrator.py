import os
from pathlib import Path
import random
import subprocess
import json
from concurrent.futures import ThreadPoolExecutor


# read all subfolders of organisms/
base_path = os.path.dirname(__file__)
generations_path = os.path.join(base_path, "generations")
all_generations = [
    f
    for f in os.listdir(generations_path)
    if os.path.isdir(os.path.join(generations_path, f))
]
latest_generation = all_generations[-1]


def get_organism_dirs_of_generation(generation: str | int) -> list[Path]:
    if isinstance(generation, int):
        generation = str(generation)
    generation_path = os.path.join(generations_path, generation, "organisms")
    # should return the actual directory objects
    return [
        Path(os.path.join(generation_path, f))
        for f in os.listdir(generation_path)
        if os.path.isdir(os.path.join(generation_path, f))
    ]


# Get one organism's phenotype
def express_organism(dir_path: Path):
    filepath = dir_path / "express_phenome.py"
    if not filepath.exists():
        raise FileNotFoundError(f"No express_phenome.py found in {dir_path}")
    # run the python file and capture the stdout
    try:
        result = subprocess.run(
            ["python", str(filepath)], check=True, capture_output=True, text=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error executing express_phenome.py: {e.stderr}")
        return ""


def select_mate(organism_dir: Path, mates: dict[str, str]):
    filepath = organism_dir / "select_mate.py"
    if not filepath.exists():
        raise FileNotFoundError(f"No pick_mate_gene.py found in {organism_dir}")
    try:
        result = subprocess.run(
            ["python", str(filepath), json.dumps(mates)],
            check=True,
            capture_output=True,
            text=True,
            input=json.dumps(mates),
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error executing select_mate.py: {e.stderr}")
        return None


# Express all organisms in a generation
def express_generation(generation: str | int):
    expressions = {}
    organism_dirs = get_organism_dirs_of_generation(generation)

    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(express_organism, organism_dir): organism_dir
            for organism_dir in organism_dirs
        }
        for future in futures:
            organism_dir = futures[future]

            with open(organism_dir / "genome.json", "r") as f:
                name = json.load(f)["name"]
            expressions[name] = {"phenotype": future.result(), "dir": organism_dir}
    return expressions


def make_kid(genome_1: dict, genome_2: dict) -> dict:
    # load the mate1 and mate2
    # for each gene in prompt_genes
    prompt_genes_1 = genome_1["prompt_genes"]
    prompt_genes_2 = genome_2["prompt_genes"]

    if len(prompt_genes_1) != len(prompt_genes_2):
        raise ValueError("Mates must have same number of prompt genes")
    output_prompt_genes = []
    for gene1, gene2 in zip(prompt_genes_1, prompt_genes_2):
        # make selection
        output_prompt_genes.append(gene1 if random.randint(0, 1) == 0 else gene2)

    parameters_1 = genome_1["parameters"]
    parameters_2 = genome_2["parameters"]
    parameters = {}
    for key in parameters_1.keys():
        parameters[key] = (
            parameters_1[key] if random.randint(0, 1) == 0 else parameters_2[key]
        )

    expression_gene_1 = genome_1["expression_gene"]
    expression_gene_2 = genome_2["expression_gene"]
    expression_gene = (
        expression_gene_1 if random.randint(0, 1) == 0 else expression_gene_2
    )
    pick_mate_gene_1 = genome_1["pick_mate_gene"]
    pick_mate_gene_2 = genome_2["pick_mate_gene"]
    pick_mate_gene = pick_mate_gene_1 if random.randint(0, 1) == 0 else pick_mate_gene_2

    name1 = genome_1["name"]
    name2 = genome_2["name"]
    name = name1 if random.randint(0, 1) == 0 else name2

    return {
        "name": name,
        "prompt_genes": output_prompt_genes,
        "parameters": parameters,
        "expression_gene": expression_gene,
        "pick_mate_gene": pick_mate_gene,
    }


def run_generation_selections(expessions: dict[str, dict[str, str | Path]]):
    selections = {}
    with ThreadPoolExecutor() as executor:
        futures = {}
        for name, data in expessions.items():
            # get other organisms
            other_organisms = {
                other_name: other_data["phenotype"]
                for other_name, other_data in expessions.items()
                if other_name != name
            }

            futures[name] = executor.submit(select_mate, data["dir"], other_organisms)

        for name, future in futures.items():
            selections[name] = {
                "selection": future.result(),
            }

    return selections


def find_matches(selections: dict[str, dict[str, str]]) -> list[tuple[str, str]]:
    matches = []
    for name, data in selections.items():
        selection = data["selection"]
        if selection == "":
            continue
        if selection in selections and selections[selection]["selection"] == name:
            match = tuple(sorted([name, selection]))
            if match not in matches:
                matches.append(match)
    return matches


def make_name_path_map(generation: str | int) -> dict[str, Path]:
    organism_dirs = get_organism_dirs_of_generation(generation)
    genome_paths = [organism_dir / "genome.json" for organism_dir in organism_dirs]
    genomes = [json.load(open(genome_path)) for genome_path in genome_paths]
    return {
        genome["name"]: organism_dir
        for organism_dir, genome in zip(organism_dirs, genomes)
    }


def get_generation_matches(generation: str | int) -> list[tuple[str, str]]:
    expressions = express_generation(generation)
    for name, data in expressions.items():
        print(f"{name}: {data['phenotype']}")
        print()
    selections = run_generation_selections(expressions)
    for name, data in selections.items():
        print(f"{name}: {data['selection']}")
        print()
    matches = find_matches(selections)
    return {
        "expressions": expressions,
        "selections": selections,
        "matches": matches,
    }


if __name__ == "__main__":
    data = get_generation_matches(latest_generation)
    expressions = data["expressions"]
    selections = data["selections"]
    matches = data["matches"]
    name_path_map = make_name_path_map(latest_generation)
    # get the organism dirs of the matches
    for match in matches:
        name_1, name_2 = match
        organism_dir_1 = name_path_map[name_1]
        organism_dir_2 = name_path_map[name_2]
        print(f"{name_1} and {name_2} are a match")
        print(f"{name_1}: {organism_dir_1}")
        print(f"{name_2}: {organism_dir_2}")
        print()
        genome_1 = json.load(open(organism_dir_1 / "genome.json"))
        genome_2 = json.load(open(organism_dir_2 / "genome.json"))
        kid = make_kid(genome_1, genome_2)
        for key, value in kid.items():
            if isinstance(value, list):
                print(f"{key}:")
                for gene in value:
                    print(f"  {gene}")
            else:
                print(f"{key}: {value}")
        print()
    # organism_dirs = [name_path_map[name] for name in matches]
    # # get genomes
    # genome_paths = [directory / "genome.json" for directory in organism_dirs]
    # genomes = [json.load(open(path)) for path in genome_paths]
    # # make kids
    # kids = [make_kid(genome_1, genome_2) for genome_1, genome_2 in zip(genomes, genomes[1:])]
    # print(kids)
    # output these to generations/0/output.json
    # turn paths into strings
    for name, data in expressions.items():
        data["dir"] = str(data["dir"])
    with open(
        os.path.join(generations_path, latest_generation, "selections.json"), "w"
    ) as f:
        json.dump(selections, f)
    with open(
        os.path.join(generations_path, latest_generation, "expressions.json"), "w"
    ) as f:
        json.dump(expressions, f)
    with open(
        os.path.join(generations_path, latest_generation, "matches.json"), "w"
    ) as f:
        json.dump(matches, f)
    # for name, data in express_generation(latest_generation).items():
    #     print(f"{name}: {data['phenotype']}")
    #     print(f"{name}: {data['dir']}")
    #     print()
