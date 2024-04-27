import os

# read all subfolders of organisms/

base_path = os.path.dirname(__file__)
organisms_path = os.path.join(base_path, "organisms")
organism_dirs = [f for f in os.listdir(organisms_path) if os.path.isdir(os.path.join(organisms_path, f))]
print(organism_dirs)

