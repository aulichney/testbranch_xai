from pathlib import Path
import json
import numpy as np
import pandas as pd
import logging
from modules.utils import create_logger

import torch
from torch.nn import functional as F
from torchvision.datasets.folder import find_classes
import pytorch_lightning as pl

import scipy as sp
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import jaccard_score, normalized_mutual_info_score

from modules.models.loaders import build_classification_loaders, build_transforms
from modules.models.classifiers import CNN
from modules.activations.activations_pytorch import *
from modules.xai.ECLAD import ECLAD
from modules.xai.ACE import ACE
from modules.xai.ConceptShap import ConceptShap

concept_extraction = {
    "ace":ACE,
    "eclad":ECLAD,
    "cshap":ConceptShap
}

def associate(dataset_config,ce,analysis, logger=None):
    if analysis=="cshap": ce.n_clusters = ce.n_concepts
    if logger is None: logger=create_logger(name="association")
    logger.info(f"Image evaluation start.")
    comparisons = []
    for class_idx, class_name in enumerate(dataset_config["classes"]):
        # get list of primitives
        primitive_paths = sorted([p for p in (Path(dataset_config["components"])/class_name).glob("*") if p.suffix in [".jpg", ".jpeg", ".png", ".bmp", ".gif"]])
        for mask_idx, masks_path in enumerate(primitive_paths):
            iter_classes = [(i*100,v) for i,v in enumerate(dataset_config["classes"])] if analysis=="ace" else [(0,"all")]
            for iter_class_idx, iter_class_name in iter_classes: # some methods, like ace, have per class concepts....
                # mask concepts
                image_path = (Path(dataset_config["base_path"])/class_name/masks_path.name.replace("_mask","")).as_posix()
                logger.info(f"eval: C[{iter_class_name}] {mask_idx} - {image_path}.")
                if analysis=="ace":
                    masks = ce.mask(image=image_path, class_name=class_name)
                elif analysis=="eclad":
                    masks = ce.mask(image=image_path)
                elif analysis=="cshap":
                    masks = ce.mask(image=image_path)
                # open primitives
                mask_primitives = np.array(Image.open(masks_path.as_posix()).convert('L').resize(ce.shape, Image.NEAREST))
                for p in np.unique(mask_primitives):
                    for c in range(ce.n_clusters):
                        image_comparison = {
                            "image":image_path,
                            "p": p,
                            "c": c,
                            "name":f"c_{int(c):02}" if analysis!="ace" else f"c_{int(c):02}_{iter_class_name}"
                        }
                        m_p = mask_primitives==p

                        if analysis=="ace":#isinstance(masks,list):
                            m_c = np.sum([mask==c for mask in masks], axis=0)>0
                        elif analysis=="eclad":
                            m_c = masks==c
                        elif analysis=="cshap":
                            m_c = masks[c]
                        # compute intersection
                        I = np.sum(m_c*m_p)
                        U = np.sum((m_c+m_p)>0)
                        # compute distances to each masks
                        edt_nc = sp.ndimage.distance_transform_edt(m_c!=True)
                        edt_np = sp.ndimage.distance_transform_edt(m_p!=True)
                        # compute distance from one mask to another
                        dst_c2p = np.sum(m_c*edt_np)
                        dst_p2c = np.sum(m_p*edt_nc)
                        sym_dst = dst_c2p+dst_p2c
                        n_c = np.sum(m_c)
                        n_p = np.sum(m_p)

                        image_comparison["n_c"]=n_c
                        image_comparison["n_p"]=n_p
                        image_comparison["dst_c2p"] = dst_c2p
                        image_comparison["dst_p2c"] = dst_p2c
                        image_comparison["sym_dst"] = sym_dst
                        image_comparison["I"] = I
                        image_comparison["U"] = U
                        image_comparison["adjusted_rand_score"] = adjusted_rand_score(m_c.flatten(), m_p.flatten())
                        image_comparison["jaccard_score"] = jaccard_score(m_c.flatten(), m_p.flatten())
                        image_comparison["normalized_mutual_info_score"] = normalized_mutual_info_score(m_c.flatten(), m_p.flatten())

                        comparisons.append(image_comparison)
    df = pd.DataFrame(comparisons)
    # consolidate
    logger.info(f"Consolidate.")
    consolidated = []
    for p in df["p"].unique():
        for c_name in df["name"].unique():
            c = df[df["name"]==c_name]["c"].values[0]
            dft = df[(df["p"]==p)&(df["name"]==c_name)]
            data = {
                "p":int(p),
                "c":int(c),
                "name":c_name,
                "dst_c2p": float(dft["dst_c2p"].mean()),
                "dst_p2c": float(dft["dst_p2c"].mean()),
                "sym_dst": float(dft["sym_dst"].mean()),
                "I": float(dft["I"].mean()),
                "U": float(dft["U"].mean()),
                "n_c": float(dft["n_c"].mean()),
                "n_p": float(dft["n_p"].mean()),
                "adjusted_rand_score": float(dft["adjusted_rand_score"].mean()),
                "jaccard_score": float(dft["jaccard_score"].mean()),
                "normalized_mutual_info_score": float(dft["normalized_mutual_info_score"].mean()),
                "ratio": len(df[(df["p"]==p)&(df["name"]==c_name)&(df["n_c"]!=0)&(df["n_p"]!=0)])/len(df[((df["p"]==p)&(df["name"]==c_name))&((df["n_c"]!=0)|(df["n_p"]!=0))])
            }
            consolidated.append(data)
    return df, consolidated

def associate_CE(
    path_model = "./data/models/AB_resnet18_plateau_0",
    path_dataset_config = "./data/configs/dataset_AB.json",
    path_training_config = None,
    path_checkpoint = None,
    path_analysis = "./data/results/ace_AB_resnet18_plateau_0",
    path_output = "./data/association/ace_AB_resnet18_plateau_0",
    seed=0,
    batch_size=32,
    force=False):
    Path(path_output).mkdir(parents=True, exist_ok=True)
    
    logger=create_logger(name="association")
    logger.info(f"Starting association: {path_output}.")
    
    # resolve configs
    if path_dataset_config is None: path_dataset_config = list(Path(path_model).glob("dataset_*.json"))[0].as_posix()
    if path_training_config is None: path_training_config = list(Path(path_model).glob("training_*.json"))[0].as_posix()
    if path_checkpoint is None: path_checkpoint = list(Path(path_model).glob("*.ckpt"))[0].as_posix()
    path_model_config = list(Path(path_model).glob("model_*.json"))[0].as_posix()
    if len(list(Path(path_analysis).glob("eclad_*.json")))>0:
        analysis="eclad"
        path_analysis_config = list(Path(path_analysis).glob("eclad_*.json"))[0].as_posix()
    elif len(list(Path(path_analysis).glob("cshap_*.json")))>0:
        analysis="cshap"
        path_analysis_config = list(Path(path_analysis).glob("cshap_*.json"))[0].as_posix()
    elif len(list(Path(path_analysis).glob("ace_*.json")))>0:
        analysis="ace"
        path_analysis_config = list(Path(path_analysis).glob("ace_*.json"))[0].as_posix()
        
    # load configs
    with open(Path(path_dataset_config).as_posix(), "r") as f:
        dataset_config = json.load(f)
    with open(Path(path_training_config).as_posix(), "r") as f:
        training_config = json.load(f)
    with open(Path(path_model_config).as_posix(), "r") as f:
        model_config = json.load(f)
    with open(Path(path_analysis_config).as_posix(), "r") as f:
        analysis_config = json.load(f)
    
    # set seed
    pl.seed_everything(seed, workers=True)
    rng = np.random.default_rng(seed)
    device = "cuda" if (analysis_config["device"]=="cuda") and torch.cuda.is_available() else "cpu"

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
    # load analysis method
    if analysis!="cshap":
        ce = concept_extraction[analysis](
            save_path=path_analysis, 
            model=model,
            transforms=transforms
        )
    else:
        train_loader, val_loader = build_classification_loaders(
            dataset_base_path = dataset_config["base_path"], 
            dataloader_config = training_config["dataloader_config"], 
            transform_train_config = training_config["transform_train_config"], 
            transform_val_config = training_config["transform_val_config"], 
            train_val_split = training_config["train_val_split"], 
            size_multiplier = 1.0,
            shuffle = training_config["shuffle"], 
            rng=rng)
        ce = concept_extraction[analysis](
            save_path=path_analysis, 
            model=model,
            transforms=transforms,
            train_loader=train_loader, 
            val_loader=val_loader
        )
    ce.load()
    ce.device = device
    if batch_size is not None: ce.batch_size = batch_size
    
    # associate
    if not (Path(path_output)/"association_full.csv").exists() or force: 
        df, consolidated = associate(dataset_config,ce,analysis, logger)
        #consolidated = [{k:float(v) if not np.isnan(v) else None for k,v in c.items()} for c in consolidated]
        
        # save configs
        with open((Path(path_output)/Path(path_model_config).name).as_posix(), "w") as f:
            json.dump(model_config, f)
        with open((Path(path_output)/Path(path_dataset_config).name).as_posix(), "w") as f:
            json.dump(dataset_config, f)
        with open((Path(path_output)/Path(path_training_config).name).as_posix(), "w") as f:
            json.dump(training_config, f)
        with open((Path(path_output)/Path(path_analysis_config).name).as_posix(), "w") as f:
            json.dump(analysis_config, f)
        
        with open((Path(path_output)/"association.json").as_posix(), "w") as f:
            json.dump(consolidated, f)
        
        df.to_csv((Path(path_output)/"association_full.csv").as_posix(), sep=";")
    logger.info("Finished.")