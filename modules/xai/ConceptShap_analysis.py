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
from .ConceptShap import ConceptShap



def cshap_analysis(
    path_model = "./data/models/AB_resnet18_plateau_0",
    path_dataset_config = None,
    path_training_config = None,
    path_checkpoint = None,
    path_output = "./data/results/cshap_AB_resnet18_plateau_0",
    path_cshap_config = "./data/configs/cshap_default.json",
    seed=0):
    Path(path_output).mkdir(parents=True, exist_ok=True)
    
    logger=create_logger(name="cshap")
    logger.info(f"Starting cshap: {path_output}.")
    
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
    with open(Path(path_cshap_config).as_posix(), "r") as f:
        cshap_config = json.load(f)
    
    # set seed
    pl.seed_everything(seed, workers=True)
    rng = np.random.default_rng(seed)
    device = "cuda" if (cshap_config["device"]=="cuda") and torch.cuda.is_available() else "cpu"

    # load transforms
    transforms = build_transforms(**training_config["transform_val_config"])

    # create dataloaders
    if cshap_config["batch_size"] is not None: training_config["dataloader_config"]["batch_size"] = cshap_config["batch_size"]
    train_loader, val_loader = build_classification_loaders(
        dataset_base_path = dataset_config["base_path"], 
        dataloader_config = training_config["dataloader_config"], 
        transform_train_config = training_config["transform_train_config"], 
        transform_val_config = training_config["transform_val_config"], 
        train_val_split = training_config["train_val_split"], 
        size_multiplier = 1.0,
        shuffle = training_config["shuffle"], 
        rng=rng)
    
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
    if "cshap_layer" not in cshap_config: cshap_config["cshap_layer"] = model_config["cshap_layer"]
    if "shape" not in cshap_config: cshap_config["shape"] = training_config["transform_val_config"]["config_list"][0]["size"]
    # run eclad
    cshap = ConceptShap( 
        save_path=path_output,
        model=model,
        transforms=transforms,
        train_loader=train_loader, 
        val_loader=val_loader,
        class_to_idx=class_to_idx,
        class_paths=class_paths,
        shape=cshap_config["shape"],
        n_limit_images=cshap_config["n_limit_images"],

        n_concepts=cshap_config["n_concepts"],
        n_epochs=cshap_config["n_epochs"],
        cshap_layer=cshap_config["cshap_layer"],
        B_threshold=cshap_config["B_threshold"],
        cshap_optim_lr=cshap_config["cshap_optim_lr"],
        lambda_1=cshap_config["lambda_1"],
        lambda_2=cshap_config["lambda_2"],
        MC_shapely_samples=cshap_config["MC_shapely_samples"],
        completeness_tune_epochs=cshap_config["completeness_tune_epochs"],
        cshap_alpha=cshap_config["cshap_alpha"],

        workers_activations=cshap_config["workers_activations"],
        batch_size=cshap_config["batch_size"],
        device=device,
        n_concept_examples=cshap_config["n_concept_examples"],
        logger=None,
        rng=rng
        )
    cshap()
    # save configs
    with open((Path(path_output)/Path(path_model_config).name).as_posix(), "w") as f:
        json.dump(model_config, f)
    with open((Path(path_output)/Path(path_dataset_config).name).as_posix(), "w") as f:
        json.dump(dataset_config, f)
    with open((Path(path_output)/Path(path_training_config).name).as_posix(), "w") as f:
        json.dump(training_config, f)
    with open((Path(path_output)/Path(path_cshap_config).name).as_posix(), "w") as f:
        json.dump(cshap_config, f)
    logger.info("Finished.")
    