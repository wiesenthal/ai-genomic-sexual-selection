import os
from pathlib import Path
import random
import subprocess
import json
from concurrent.futures import ThreadPoolExecutor
import random
import string
from mutate import mutate_or_keep_val
import names
from image_generator import generate_images_from_generation

all_names_ever = set()

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


def print_dict(obj: dict):
    if isinstance(obj, str):
        print(obj)
        return
    if isinstance(obj, list):
        for item in obj:
            print_dict(item)
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            print(f"{key}: ")
            print("  ", end="")
            print_dict(value)
            print()
            return


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
        with open(dir_path / "errors.log", "a") as error_file:
            error_file.write(
                f"\n{str(dir_path)}\nError executing express_phenome.py: {e.stderr}\n"
            )
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
        with open(organism_dir / "errors.log", "a") as error_file:
            error_file.write(
                f"\n{str(organism_dir)}\nError executing select_mate.py: {e.stderr}\n"
            )
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


def make_random_value(length: int) -> str:
    choices = (
        string.digits + string.ascii_letters
    )  # Combines digits, lowercase and uppercase letters
    random_value = "".join(random.choice(choices) for _ in range(length))
    return random_value


def make_kid(path_1: Path, path_2: Path) -> dict:
    with open(path_1 / "genome.json", "r") as f:
        genome_1 = json.load(f)
    with open(path_2 / "genome.json", "r") as f:
        genome_2 = json.load(f)
    with open(path_1 / "complete.py", "r") as f:
        complete_1 = f.read()
    with open(path_2 / "complete.py", "r") as f:
        complete_2 = f.read()
    with open(path_1 / "express_phenome.py", "r") as f:
        express_phenome_1 = f.read()
    with open(path_2 / "express_phenome.py", "r") as f:
        express_phenome_2 = f.read()
    with open(path_1 / "select_mate.py", "r") as f:
        select_mate_1 = f.read()
    with open(path_2 / "select_mate.py", "r") as f:
        select_mate_2 = f.read()
    # load the mate1 and mate2
    # for each gene in prompt_genes
    prompt_genes_1 = genome_1["prompt_genes"]
    prompt_genes_2 = genome_2["prompt_genes"]

    if len(prompt_genes_1) != len(prompt_genes_2):
        raise ValueError("Mates must have same number of prompt genes")

    parameters_1 = genome_1["parameters"]
    parameters_2 = genome_2["parameters"]
    parameters = {}
    for key in parameters_1.keys():
        val = parameters_1[key] if random.randint(0, 1) == 0 else parameters_2[key]
        parameters[key] = val
    mutation_rate = parameters["mutation_rate"]
    mutation_prompt = parameters["mutation_prompt"]
    mutation_rate = mutate_or_keep_val(mutation_rate, mutation_prompt, mutation_rate)
    mutation_prompt = mutate_or_keep_val(
        mutation_prompt, mutation_prompt, mutation_rate
    )
    parameters["mutation_prompt"] = mutation_prompt
    parameters["mutation_rate"] = float(mutation_rate)
    

    output_prompt_genes = []
    for gene1, gene2 in zip(prompt_genes_1, prompt_genes_2):
        # make selection
        gene = gene1 if random.randint(0, 1) == 0 else gene2
        gene = mutate_or_keep_val(gene, mutation_prompt, mutation_rate)
        output_prompt_genes.append(gene)

    expression_gene_1 = genome_1["expression_gene"]
    expression_gene_2 = genome_2["expression_gene"]
    expression_gene = (
        expression_gene_1 if random.randint(0, 1) == 0 else expression_gene_2
    )

    expression_gene = mutate_or_keep_val(
        expression_gene, mutation_prompt, mutation_rate
    )

    pick_mate_gene_1 = genome_1["pick_mate_gene"]
    pick_mate_gene_2 = genome_2["pick_mate_gene"]
    pick_mate_gene = pick_mate_gene_1 if random.randint(0, 1) == 0 else pick_mate_gene_2

    pick_mate_gene = mutate_or_keep_val(pick_mate_gene, mutation_prompt, mutation_rate)

    name = names.get_full_name()
    while name in all_names_ever:
        name = names.get_full_name()
    all_names_ever.add(name)

    complete = complete_1 if random.randint(0, 1) == 0 else complete_2
    express_phenome = (
        express_phenome_1 if random.randint(0, 1) == 0 else express_phenome_2
    )
    select_mate = select_mate_1 if random.randint(0, 1) == 0 else select_mate_2

    return {
        "genome": {
            "name": name,
            "prompt_genes": output_prompt_genes,
            "parameters": parameters,
            "expression_gene": expression_gene,
            "pick_mate_gene": pick_mate_gene,
        },
        "complete": complete,
        "express_phenome": express_phenome,
        "select_mate": select_mate,
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


def match_generation_iteration(
    expressions: dict[str, dict[str, str | Path]],
) -> dict[str, dict[str, str | Path]]:
    selections = run_generation_selections(expressions)
    matches = find_matches(selections)
    return {
        "expressions": expressions,
        "selections": selections,
        "matches": matches,
    }


def get_generation_matches(
    expressions: dict[str, dict[str, str | Path]],
    population_size: int,
    max_attempts: int,
) -> list[tuple[str, str]]:
    matches = []
    attempts = 0
    while len(matches) < population_size and attempts < max_attempts:
        print(f"Selection step {attempts + 1}")
        data = match_generation_iteration(expressions)
        matches.extend(data["matches"])
        attempts += 1
        # make selections dir
        selections_dir = os.path.join(
            generations_path, latest_generation, "outputs", "selections"
        )
        os.makedirs(selections_dir, exist_ok=True)
        with open(
            os.path.join(selections_dir, f"{attempts}.json"),
            "w",
        ) as f:
            json.dump(data["selections"], f)
    return matches


def make_kids(matches, name_path_map):
    kids = []
    for match in matches:
        name_1, name_2 = match
        organism_dir_1 = name_path_map[name_1]
        organism_dir_2 = name_path_map[name_2]
        kid = make_kid(organism_dir_1, organism_dir_2)
        kids.append(kid)
    return kids


def run_generation(generation: str | int):
    expressions = express_generation(generation)
    matches = get_generation_matches(expressions, 12, 12)
    print(f"Matches: {len(matches)} - {matches}")
    name_path_map = make_name_path_map(generation)
    # get the organism dirs of the matches
    kids = make_kids(matches, name_path_map)
    matches_and_kids_dicts = [
        {"match": match, "child_genome": kid["genome"]}
        for match, kid in zip(matches, kids)
    ]
    outputs_dir = os.path.join(generations_path, generation, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    with open(os.path.join(outputs_dir, "matches_and_kids.json"), "w") as f:
        json.dump(matches_and_kids_dicts, f)
    for name, data in expressions.items():
        data["dir"] = str(data["dir"])
    with open(os.path.join(outputs_dir, "expressions.json"), "w") as f:
        json.dump(expressions, f)
    with open(os.path.join(outputs_dir, "matches.json"), "w") as f:
        json.dump(matches, f)
    return {
        "kids": kids,
        "matches": matches,
        "expressions": expressions,
    }


from concurrent.futures import ThreadPoolExecutor
import time

# Global executor
executor = ThreadPoolExecutor()


def generate_images_with_backoff(expressions, generation, attempt=1):
    try:
        generate_images_from_generation(expressions, generation)
    except Exception as e:
        if attempt <= 5:  # Limit the number of retries
            print(f"Attempt {attempt} failed: {e}")
            time.sleep(61)  # Exponential backoff
            generate_images_with_backoff(expressions, generation, attempt + 1)
        else:
            outputs_dir = os.path.join(generations_path, generation, "outputs")
            with open(os.path.join(outputs_dir, "errors.log"), "a") as error_file:
                error_file.write(
                    f"Error generating images for generation {generation} after {attempt} attempts: {e}\n"
                )


def try_generate_images(
    generation: str | int, expressions: dict[str, dict[str, str | Path]]
):
    # Submit the task to the global executor
    executor.submit(generate_images_with_backoff, expressions, generation)


# Ensure to shutdown the executor properly when the application is closing
import atexit

atexit.register(executor.shutdown)


def loop():
    generation = "67"
    while True:
        print("~~~")
        print(f"Running generation {generation}")
        print()
        data = run_generation(generation)
        matches = data["matches"]
        kids = data["kids"]
        if len(kids) == 0:
            print("No kids found, stopping. EXTINCTION.")
            break
        expressions = data["expressions"]
        # generate images
        try_generate_images(generation, expressions)
        # make the next generation
        generation = str(int(generation) + 1)

        # make a new dir in generations
        next_organisms_dir = os.path.join(generations_path, generation, "organisms")
        os.makedirs(next_organisms_dir, exist_ok=True)
        for kid in kids:
            kid_dir = os.path.join(next_organisms_dir, kid["genome"]["name"])
            os.makedirs(kid_dir, exist_ok=True)
            with open(os.path.join(kid_dir, "genome.json"), "w") as f:
                json.dump(kid["genome"], f)
            with open(os.path.join(kid_dir, "complete.py"), "w") as f:
                f.write(kid["complete"])
            with open(os.path.join(kid_dir, "express_phenome.py"), "w") as f:
                f.write(kid["express_phenome"])
            with open(os.path.join(kid_dir, "select_mate.py"), "w") as f:
                f.write(kid["select_mate"])


if __name__ == "__main__":
    loop()
