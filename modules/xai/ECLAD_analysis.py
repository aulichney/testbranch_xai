from pathlib import Path
import json
import numpy as np
import logging
from modules.utils import create_logger

import torch
from torch.nn import functional as F
from torchvision.datasets.folder import find_classes
import pytorch_lightning as pl

from modules.models.loaders import build_classification_loaders, build_transforms
from modules.models.classifiers import CNN
from modules.activations.activations_pytorch import *
from .ECLAD import ECLAD



def eclad_analysis(
    path_model = "./data/models/AB_resnet18_plateau_0",
    path_dataset_config = None,
    path_training_config = None,
    path_checkpoint = None,
    path_output = "./data/results/eclad_AB_resnet18_plateau_0_n10s",
    path_eclad_config = "./data/configs/eclad_n10s.json",
    seed=0):
    Path(path_output).mkdir(parents=True, exist_ok=True)
    
    logger=create_logger(name="eclad")
    logger.info(f"Starting eclad: {path_output}.")
    
    # resolve configs
    if path_dataset_config is None: path_dataset_config = list(Path(path_model).glob("dataset_*.json"))[0].as_posix()
    if path_training_config is None: path_training_config = list(Path(path_model).glob("training_*.json"))[0].as_posix()
    if path_checkpoint is None: path_checkpoint = list(Path(path_model).glob("*.ckpt"))[0].as_posix()
    path_model_config = list(Path(path_model).glob("model_*.json"))[0].as_posix()
    
    # load configs
    with open(Path(path_dataset_config).as_posix(), "r") as f:
        dataset_config = json.load(f)
    with open(Path(path_training_config).as_posix(), "r") as f:
        training_config = json.load(f)
    with open(Path(path_model_config).as_posix(), "r") as f:
        model_config = json.load(f)
    with open(Path(path_eclad_config).as_posix(), "r") as f:
        eclad_config = json.load(f)
    
    # set seed
    pl.seed_everything(seed, workers=True)
    rng = np.random.default_rng(seed)
    device = "cuda" if (eclad_config["device"]=="cuda") and torch.cuda.is_available() else "cpu"

    # load transforms
    transforms = build_transforms(**training_config["transform_val_config"])
    
    # load model
    classes, class_to_idx = find_classes(dataset_config["base_path"])
    class_paths={k:(Path(dataset_config["base_path"])/str(k)).as_posix() for k in class_to_idx.keys()}
    model_config["num_classes"]=len(classes)
    model = CNN(
        model_name=model_config["model_name"], 
        pretrained=model_config["pretrained"], 
        num_classes=model_config["num_classes"], 
        in_chans=model_config["in_chans"],
    )
    checkpoint = torch.load(path_checkpoint)
    model.load_state_dict(checkpoint)
        
    def correct_inplace_relu(model):
        for child_name, child in model.named_children():
            if isinstance(child, nn.ReLU):
                setattr(model, child_name, nn.ReLU())
            else:
                correct_inplace_relu(child)
    correct_inplace_relu(model)
    
    model.eval()
    model.to(device)

    # update config
    if "eclad_layers" not in eclad_config: eclad_config["eclad_layers"] = model_config["eclad_layers"]
    if "shape" not in eclad_config: eclad_config["shape"] = training_config["transform_val_config"]["config_list"][0]["size"]
    # run eclad
    eclad = ECLAD( 
        save_path=path_output,
        model=model,
        transforms=transforms,
        class_to_idx=class_to_idx,
        class_paths=class_paths,
        shape=eclad_config["shape"],
        n_limit_images=eclad_config["n_limit_images"],
        n_clusters=eclad_config["n_clusters"],
        eclad_layers=eclad_config["eclad_layers"],
        upsampling_mode=eclad_config["upsampling_mode"],
        clustering_method=eclad_config["clustering_method"],
        eclad_clustering_batches=eclad_config["eclad_clustering_batches"],
        eclad_sample=eclad_config["eclad_sample"],
        eclad_alpha=eclad_config["eclad_alpha"],
        eclad_standardscale=eclad_config["eclad_standardscale"],
        workers_activations=eclad_config["workers_activations"],
        batch_size=eclad_config["batch_size"],
        device=device,
        n_concept_examples=eclad_config["n_concept_examples"],
        logger=None,
        rng=rng
        )
    eclad()
    # save configs
    with open((Path(path_output)/Path(path_model_config).name).as_posix(), "w") as f:
        json.dump(model_config, f)
    with open((Path(path_output)/Path(path_dataset_config).name).as_posix(), "w") as f:
        json.dump(dataset_config, f)
    with open((Path(path_output)/Path(path_training_config).name).as_posix(), "w") as f:
        json.dump(training_config, f)
    with open((Path(path_output)/Path(path_eclad_config).name).as_posix(), "w") as f:
        json.dump(eclad_config, f)
    logger.info("Finished.")
    