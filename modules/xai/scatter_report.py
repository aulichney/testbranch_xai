from pathlib import Path
import json
import numpy as np
import pandas as pd
import logging
from modules.utils import create_logger

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import seaborn as sns
import copy
from PIL import Image

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
import matplotlib
#matplotlib.use('agg')
concept_extraction = {
    "ace":ACE,
    "eclad":ECLAD,
    "cshap":ConceptShap
}


def scatterplot_report_CE(
    path_model = "./data/models/AB_resnet18_plateau_0",
    path_dataset_config = "./data/configs/dataset_AB.json",
    path_training_config = None,
    path_checkpoint = None,
    path_analysis = 'data/results/eclad_AB_resnet18_plateau_0_n10s',
    path_association = 'data/association/eclad_AB_resnet18_plateau_0_n10s',
    path_output = "./data/reports/eclad_AB_resnet18_plateau_0_n10s",
    seed=0,
    force=False,
    distance="sym_dst", scatter_xlim = 3e6):
    Path(path_output).mkdir(parents=True, exist_ok=True)
    logger=create_logger(name="report")
    logger.info(f"Starting reporting: {path_output}.")
    
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
    
    
    name = Path(path_analysis).name
    method = analysis_config["type"]
    model = model_config["type"]
    dataset = dataset_config["name"]

    # load importance
    with open((Path(path_analysis)/"results.json").as_posix(), "rb") as f:
        results = json.load(f)
        if method=="ace": importance = [{"c":r["idx"], "name":r["name"], "importance":(r["score_mean"]*2-1), "concept_paths":r["concept_path"]} for r in results["results"]]
        if method=="eclad": importance = [{"c":r["idx"], "name":r["name"], "importance":r["RI"], "concept_paths":r["concept_paths"]} for r in results["results"]]
        if method=="cshap": importance = [{"c":r["idx"], "name":r["name"], "importance":r["shap_mean"], "concept_paths":r["concept_paths"]} for r in results["results"]]

    # load association
    if not (Path(path_association)/"association.json").exists(): return
    with open((Path(path_association)/"association.json").as_posix(), "rb") as f:
        consolidated = json.load(f)

    # consolidate
    consolidated = [{
        **c, 
        "importance":[i for i in importance if i["name"]==c["name"]][0]["importance"], 
        "concept_paths":[i for i in importance if i["name"]==c["name"]][0]["concept_paths"]} 
        for c in consolidated]

    dfc = pd.DataFrame(consolidated)
    df_tmp = dfc.sort_values(by=[distance], ascending=True).drop_duplicates(["name"])
    df_tmp = df_tmp[df_tmp[distance]<scatter_xlim]

    p_symbols={
            "AB":{1:"A",2:"B",3:"+",4:"bg"},
            "ABplus":{1:"A",2:"B",3:"+",4:"*",5:"/",6:"#",7:"X",8:"bg"}, 
            "CO":{1:"C",2:"O", 3:"+",4:"bg"}, 
            "BigSmall":{1:"big B",2:"small B", 3:"+", 4:"bg"}, 
            "isA":{1:"A",2:"others", 0:"bg"}, 
            "colorGB":{1:"colored", 2:"D", 3:"+", 4:"bg"}, 
            "leather":{0:"",1:"color",2:"cut",3:"fold",4:"glue",5:"good",6:"poke"}, 
            "metal_nut":{0:"",1:"bent",2:"color",3:"flip",4:"good",5:"scratch"}
        }
    if dataset in p_symbols:
        df_tmp["p"] = df_tmp["p"].map(p_symbols[dataset])
    else:
        df_tmp["p"] = df_tmp["p"].apply(str)
    n_classes = len(results["class_to_idx"])
    font_scale = 2.5
    #sns.set(font_scale = font_scale)
    n_rows = 6
    n_concepts = 6
    left_space = 1.0
    scatter_width=4.5
    spacing=0.01
    figsize = (left_space+n_classes+scatter_width, n_rows)
    left_space = 1.0*left_space/figsize[0]
    spacing_scatter=0.1/figsize[0]
    scatter_width=1.0*scatter_width/figsize[0]

    box_scatter = [1.0-scatter_width+spacing_scatter, 0.0, scatter_width-spacing_scatter, 1.0]#[left, bottom, width, height]
    plt.rcParams['figure.dpi'] = 200
    plt.rcParams['savefig.dpi'] = 200
    fig = plt.figure(figsize=figsize)
    ######### Plot scatter of the concepts
    axs = fig.add_axes(box_scatter)
    axs.axhline(linewidth=1, color="k")
    axs.axvline(linewidth=1, color="k")
    hue_order=list(p_symbols[dataset].values())
    sns.scatterplot(data=df_tmp, ax=axs,
        x=distance, y="importance",
        hue="p", #style="p",
        hue_order=hue_order,
        palette="tab10",
        s=200, linewidth=0, alpha = 0.7)
    axs.tick_params(axis='x', labelsize=10*font_scale)
    axs.tick_params(axis='y', labelsize=10*font_scale)
    for row in df_tmp.to_dict('records'):
        if row[distance]>scatter_xlim: continue
        s = row["name"].replace("c_0","").replace("c_","")
        if method=="ace":
            s = s.replace("_","-")
            for k,v in class_to_idx.items():
                s = s.replace(k,f"{v}")
        axs.text(
            x=row[distance], 
            y=row["importance"], 
            s=s, 
            fontsize=int(7*font_scale), 
            horizontalalignment='center', verticalalignment='center')
    axs.set_xlabel("Distance", fontsize=10*font_scale)
    axs.set_ylabel("Importance", fontsize=10*font_scale)
    #axs.set_title(f"{name}")
    axs.set_xlim(left=-0.04*scatter_xlim, right=scatter_xlim)#1.01)
    axs.set_ylim(-1.1,1.1)#-0.01
    axs.yaxis.tick_right()
    axs.yaxis.set_label_position("right")
    axs.grid()

    ######### plot the image examples
    b_size_h = 1.0/n_rows
    b_size_w = b_size_h * figsize[1] / figsize[0]
    # select the concepts, half are the most important, the other half the
    concet_list = df_tmp.sort_values("importance", ascending=False).to_dict('records')
    # get clossest concepts which are not the most important
    cidxs = np.argsort([c[distance] for c in concet_list][int(n_concepts/2):]).tolist()[:n_concepts-int(n_concepts/2)]
    selected_concepts = concet_list[:int(n_concepts/2)] + [concet_list[i+int(n_concepts/2)] for i in cidxs]
    img_paths = {class_name:list((Path(dataset_config["base_path"])/class_name).glob("*")) for class_name in results["class_to_idx"].keys()}
    n_samples = max([len(l) for l in img_paths.values()])
    concept_plotted = [0 for c in range(min(len(selected_concepts), n_concepts))]
    masks_dict = {}
    m_c_dict = {}
    img_dict = {}
    for img_idx in range(n_samples):
        # evaluate images
        for class_name_rep, class_idx_rep in results["class_to_idx"].items():
            if img_idx < len(img_paths[class_name_rep]):
                img_path = img_paths[class_name_rep][img_idx].as_posix()
                img = Image.open(img_path).resize(ce.shape)
                ### mask image
                if analysis=="ace":
                    masks = ce.mask(image=img_path, class_name=class_name_rep)
                elif analysis=="eclad":
                    masks = ce.mask(image=img_path)
                elif analysis=="cshap":
                    masks = ce.mask(image=img_path)
                img_dict[class_name_rep] = img.copy()
                masks_dict[class_name_rep] = masks.copy()
        # search and plot concept
        for idx in range(min(len(selected_concepts), n_concepts)):
            i = n_rows - idx -1
            c = selected_concepts[idx]
            # search concepts in images
            for class_name_rep, class_idx_rep in results["class_to_idx"].items():
                if img_idx < len(img_paths[class_name_rep]):
                    ### mask concept
                    if analysis=="ace":#isinstance(masks,list):
                        m_c = np.sum([mask==c["c"] for mask in masks_dict[class_name_rep]], axis=0)>0
                    elif analysis=="eclad":
                        m_c = masks_dict[class_name_rep]==c["c"]
                    elif analysis=="cshap":
                        m_c = masks_dict[class_name_rep][c["c"]]
                    m_c_dict[class_name_rep] = m_c.copy()
            
            mask_has_concept = np.sum([m_c_dict[class_name_rep] for class_name_rep in results["class_to_idx"].keys()])
            if concept_plotted[idx]>mask_has_concept: continue
            if mask_has_concept>0:
                # show the concept in the figure
                for class_name, class_idx in results["class_to_idx"].items():
                    box= [b_size_w*class_idx+left_space, b_size_h*i, b_size_w-spacing, b_size_h-spacing]
                    ax = fig.add_axes(box)
                    ax.set_axis_off()
                    img_c = Image.fromarray(((m_c_dict[class_name][...,np.newaxis]*img_dict[class_name])+((m_c_dict[class_name][...,np.newaxis]==False)*img_dict[class_name]*0.4)).astype(np.uint8)) 
                    ax.imshow(img_c)
                    if idx==0: ax.set_title(f"{class_name}", fontsize=10*font_scale)
                    if class_idx==0: 
                        s = c['name'].replace("c_0","").replace("c_","")
                        if method=="ace":
                            s = s.replace("_","-")
                            for k,v in class_to_idx.items():
                                s = s.replace(k,f"{v}")
                        fig.text(b_size_w*class_idx, b_size_h*(i+0.5), s, fontsize=10*font_scale, verticalalignment='center')
                    elif class_idx==len(results["class_to_idx"])-1:
                        con = ConnectionPatch(xyA=(img_c._size[0],int(img_c._size[1]/2)), xyB=(c[distance], c["importance"]),
                                            coordsA="data", coordsB="data",
                                            axesA=ax, axesB=axs, color="black",
                                            arrowstyle="->", shrinkA=5, shrinkB=5,
                                            mutation_scale=10, fc="w", ls="--")
                        ax.add_artist(con)
            concept_plotted[idx] = mask_has_concept
        logger.info(f"Testing images {img_idx}. plotted {concept_plotted}")
        if all([cs>0 for cs in concept_plotted]): 
            logger.info(f"{concept_plotted}")
            break
    fname = (Path(path_output)/"scatter.png")
    fig.savefig(fname, dpi="figure", facecolor='white', bbox_inches='tight', transparent=False)
    fname = (Path(path_output)/"scatter.svg")
    fig.savefig(fname, format="svg", dpi=600, facecolor='white', bbox_inches='tight', transparent=False)
    #plt.show()
    plt.close()
    logger.info("Finished.")