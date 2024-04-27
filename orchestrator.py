import os
from pathlib import Path
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
            expressions[name] = future.result()
    return expressions


if __name__ == "__main__":
    for (k, v) in express_generation(latest_generation).items():
        print(f"{k}: {v}\n")

