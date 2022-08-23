from pathlib import Path

import json
import numpy as np
import pandas as pd
import logging
from modules.utils import create_logger

import torch
from torch.nn import functional as F
import torchvision
from torchvision.datasets.folder import find_classes
import pytorch_lightning as pl

from modules.models.loaders import build_classification_loaders, build_transforms
from modules.models.classifiers import CNN
from modules.activations.activations_pytorch import *

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from captum.attr import GuidedGradCam


def model_eval(
    path_model = "./data/models/AB_resnet18_plateau_0",
    path_dataset_config = None,
    path_training_config = None,
    path_checkpoint = None,
    path_output = "./data/models_eval/AB_resnet18_plateau_0",
    seed=0,
    device="cuda"):
    Path(path_output).mkdir(parents=True, exist_ok=True)
    
    logger=create_logger(name="space")
    logger.info(f"Starting space: {path_output}.")
    
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
    
    # set seed
    pl.seed_everything(seed, workers=True)
    rng = np.random.default_rng(seed)
    device = "cuda" if (device=="cuda") and torch.cuda.is_available() else "cpu"

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
    
    # create full dataloader
    dataset = torchvision.datasets.ImageFolder(root=dataset_config["base_path"], transform=transforms)
    dataloader = torch.utils.data.DataLoader(dataset, **{**training_config["dataloader_config"],**{"shuffle":False, 'num_workers': 0, "batch_size":12}})
    
    # evaluate dataset
    probs = []
    for batch in dataloader:
        x,y=batch
        x,y = x.to(device), y.to(device)
        logits = model(x)
        prob = F.softmax(logits, dim=-1)
        probs.extend(prob.cpu().detach().tolist())
    probs = np.array(probs)
    
    # confusion matrix
    fig, ax = plt.subplots(1,1,figsize=(len(dataset.classes)*3, len(dataset.classes)*3))
    cm = confusion_matrix(dataset.targets, probs.argmax(1), labels=range(probs.shape[1]), normalize="true")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dataset.classes)
    disp.plot(ax=ax)
    fig.savefig((Path(path_output)/"confusion_matrix.png").as_posix())
    
    # true prob plots
    fig, axs = plt.subplots(len(dataset.classes), 1, sharex="all", figsize=(20,len(dataset.classes)*2))
    for k_name, k_idx in class_to_idx.items():
        probs_true_class = probs[(np.array(dataset.targets)==k_idx),k_idx]
        sns.kdeplot(data=probs_true_class, ax=axs[k_idx], clip=[0,1], fill=True)
        sns.rugplot(data=probs_true_class, ax=axs[k_idx], expand_margins=True, lw=1, alpha=0.3, height=0.2)
        axs[k_idx].set_ylabel(k_name)
    axs[-1].set_xlabel("probability")
    fig.savefig((Path(path_output)/"true_class_probabilities.png").as_posix())
    
    # saliency
    df = pd.DataFrame({
        "y":dataset.targets,
        "pred":probs.argmax(1),
        "max_prob":probs.max(1),
        "prob_true_class":[probs[i,dataset.targets[i]] for i in range(len(probs))]
    })

    model.to(device)
    layers = get_layers(model, names=[model_config["gradcam_layer"]])
    guided_gc = GuidedGradCam(model, layers[model_config["gradcam_layer"]])

    n = 4
    fig, axs = plt.subplots(len(class_to_idx)*2, n*2, figsize = (n*2*3, len(class_to_idx)*2*3))
    for k_name, k_idx in class_to_idx.items():
        easy_idxs = df[(df["y"]==k_idx)].sort_values("prob_true_class", ascending=False).index.tolist()[:n+1]
        hard_idxs = df[(df["y"]==k_idx)].sort_values("prob_true_class", ascending=True).index.tolist()[:n+1]

        axs[k_idx*2,0].set_ylabel(k_name)
        axs[k_idx*2+1,0].set_ylabel("GuidedGradCam")
        for i in range(n):
            # easy
            img = Image.open(dataset.imgs[easy_idxs[i]][0]).convert('RGB')
            axs[k_idx*2,i].imshow(img)
            axs[k_idx*2,i].set_axis_off()
            axs[k_idx*2,i].set_title(f"y: {k_idx} prob: {df['prob_true_class'].values[easy_idxs[i]]:.02f}")

            x = transforms(img)
            x = x.to(device)
            x.requires_grad=True
            attribution = guided_gc.attribute(x.unsqueeze(0), k_idx)[0]
            attribution = F.relu(attribution).mean(0).detach().cpu().numpy()
            axs[k_idx*2+1,i].imshow(attribution)
            axs[k_idx*2+1,i].set_axis_off()

            # hard
            img = Image.open(dataset.imgs[hard_idxs[i]][0]).convert('RGB')
            axs[k_idx*2,n+i].imshow(img)
            axs[k_idx*2,n+i].set_axis_off()
            axs[k_idx*2,n+i].set_title(f"y: {k_idx} prob: {df['prob_true_class'].values[hard_idxs[i]]:.02f}")

            x = transforms(img)
            x = x.to(device)
            x.requires_grad=True
            attribution = guided_gc.attribute(x.unsqueeze(0), k_idx)[0]
            attribution = F.relu(attribution).mean(0).detach().cpu().numpy()
            axs[k_idx*2+1,n+i].imshow(attribution)
            axs[k_idx*2+1,n+i].set_axis_off()
    fig.savefig((Path(path_output)/"examples_saliency.png").as_posix())

    logger.info("Finished.")
    for h in logger.handlers:
        logger.removeHandler(h)



def models_eval(
    path_models = "./data/models",
    path_output = "./data/models_eval",
    seed=0,
    device="cuda"):
    for idx,p in enumerate(Path(path_models).glob("*")):
        print(p)
        if idx<=9: continue
        model_eval(
            path_model = p.as_posix(),
            path_output = f"{path_output}/{p.name}",
            seed=seed,
            device=device)