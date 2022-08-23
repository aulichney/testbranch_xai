from pathlib import Path
import json

def create_dataset_configs(base_path="./data/datasets", output_path="./data/configs"):
    """
    Create a json config file for each dataset present in the base_path.

    Args:
        base_path (str, optional): path where the datasets are saved. Defaults to "./data/datasets".
        output_path (str, optional): path where hte configs will be saved. Defaults to "./data/configs".
    """
    Path(output_path).mkdir(parents=True, exist_ok=True)
    for p in Path(base_path).glob("*"):
        # verify that it's a dir and it's not textures
        if not (p.is_dir() or p.name=="texture"): continue
        # create base config for the dataset
        config = {
            "type":"dataset",
            "dataset_path": p.as_posix(), 
            "base_path": (p/"train").as_posix() if (p/"train").exists() else p.as_posix(),
            "name": p.name,
            "components": (p/"components").as_posix() if (p/"components").exists() else None,
            "ground_truth": (p/"ground_truth").as_posix() if (p/"ground_truth").exists() else None,
            "classes": sorted([c.name for c in p.glob("*") if c.is_dir()]) if not (p/"train").exists() else sorted([c.name for c in (p/"train").glob("*") if c.is_dir()]),
            "dataset_type": "classification"
        }
        if "./" in base_path:
            for c in ["dataset_path", "base_path", "components", "ground_truth"]:
                if config[c] is not None:
                    config[c]= "./"+config[c]
        with open((Path(output_path)/f"dataset_{p.name}.json").as_posix(), "w") as f:
            json.dump(config, f, ensure_ascii=True, indent=4)

