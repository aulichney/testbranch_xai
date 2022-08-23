from pathlib import Path
import json

import numpy as np
import torch
from torchvision.datasets.folder import find_classes

import pytorch_lightning as pl
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint
from .loaders import build_classification_loaders
from .classifiers import CNN
import logging
from modules.utils import create_logger

def train(
    path_model_config = "./data/configs/model_resnet18.json", 
    path_dataset_config = "./data/configs/dataset_AB.json",
    path_training_config = "./data/configs/training_plateau.json",
    path_output="./data/models/AB_resnet18_plateau_0",
    seed=0,
    device=None):
    torch.use_deterministic_algorithms(True, warn_only=True)
    pl.seed_everything(seed, workers=True)
    rng = np.random.default_rng(seed)
    Path(path_output).mkdir(parents=True, exist_ok=True)

    logger=create_logger(name="training")
    logger.info(f"Starting training: {path_output}.")

    # load configs
    with open(Path(path_model_config).as_posix(), "r") as f:
        model_config = json.load(f)
    with open(Path(path_dataset_config).as_posix(), "r") as f:
        dataset_config = json.load(f)
    with open(Path(path_training_config).as_posix(), "r") as f:
        training_config = json.load(f)
    
    # create dataloaders
    train_loader, val_loader = build_classification_loaders(
        dataset_base_path = dataset_config["base_path"], 
        dataloader_config = training_config["dataloader_config"], 
        transform_train_config = training_config["transform_train_config"], 
        transform_val_config = training_config["transform_val_config"], 
        train_val_split = training_config["train_val_split"], 
        size_multiplier = training_config.get('size_multiplier', 1.0), 
        shuffle = training_config["shuffle"], 
        rng=rng)
    
    # create model
    classes, class_to_idx = find_classes(dataset_config["base_path"])
    model_config["classes"] = classes
    model_config["class_to_idx"] = class_to_idx
    model_config["num_classes"]=len(classes)
    if training_config["scheduler_config"]["type"]=="OneCycleLR": 
        training_config["scheduler_config"]["steps_per_epoch"] = 1
        training_config["scheduler_config"]["epochs"] = training_config["max_epochs"]
    model = CNN(
        model_name=model_config["model_name"], 
        pretrained=model_config["pretrained"], 
        num_classes=model_config["num_classes"], 
        in_chans=model_config["in_chans"],
        checkpoint_path=model_config["checkpoint_path"], 
        loss_name=training_config["loss_name"], 
        learning_rate=training_config["learning_rate"], 
        optimizer_config=training_config["optimizer_config"],
        scheduler_config=training_config["scheduler_config"],
        scheduler_monitor=training_config["scheduler_monitor"]
    )
    # create trainer
    if device=="cpu": gpu=None
    elif isinstance(device, int): gpu=device
    else: gpu = 1 if torch.cuda.is_available() else None
    early_stop_callback = EarlyStopping(**training_config["early_stopping_config"])
    checkpoint_callback = ModelCheckpoint(**training_config["checkpoint_saving_config"])
    progressbar_callback = TQDMProgressBar(refresh_rate=10)
    logger_tensorboard = TensorBoardLogger(path_output, name=Path(path_output).name)
    trainer = pl.Trainer(
        gradient_clip_val=training_config["gradient_clip_val"], 
        min_epochs=training_config["min_epochs"], 
        max_epochs=training_config["max_epochs"], 
        logger=logger_tensorboard,
        callbacks=[early_stop_callback, checkpoint_callback, progressbar_callback], 
        gpus=gpu, 
        auto_lr_find=training_config["auto_lr_find"],
        deterministic=True)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if training_config["auto_lr_find"]:
        trainer.tune(model, train_loader, val_loader,
            lr_find_kwargs = {
                "min_lr":training_config["auto_lr_min"],
                "max_lr":training_config["auto_lr_max"],
                "num_training":training_config["auto_lr_num"],
                "early_stop_threshold": 4 if "auto_lrearly_stop_threshold" not in training_config else training_config["auto_lrearly_stop_threshold"]
            })
    # train
    logger.info("Training.")
    trainer.fit(model, train_loader, val_loader)
    # save model
    logger.info("Saving.")
    torch.save(model.state_dict(), (Path(path_output)/"checkpoint.ckpt").as_posix())
    # save configs
    with open((Path(path_output)/Path(path_model_config).name).as_posix(), "w") as f:
        json.dump(model_config, f)
    with open((Path(path_output)/Path(path_dataset_config).name).as_posix(), "w") as f:
        json.dump(dataset_config, f)
    with open((Path(path_output)/Path(path_training_config).name).as_posix(), "w") as f:
        json.dump(training_config, f)
    logger.info("Finished.")


        