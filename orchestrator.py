import os
from pathlib import Path
import subprocess
import json

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


def run_generation(generation: str | int):
    next_generation = int(generation) + 1
    expressions = {}
    # should async this
    for organism_dir in get_organism_dirs_of_generation(generation):
        # load genome file to get name
        with open(organism_dir / "genome.json", "r") as f:
            name = json.load(f)["name"]
        expressions[name] = express_organism(organism_dir)
    return expressions


print(run_generation(latest_generation))

# organism_dirs = [f for f in os.listdir(generations_path) if os.path.isdir(os.path.join(generations_path, f))]
# print(organism_dirs)
