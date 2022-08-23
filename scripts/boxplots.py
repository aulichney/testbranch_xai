from pathlib import Path
import os
import json
import numpy as np
import pandas as pd
import time

import matplotlib.pyplot as plt
import seaborn as sns
import copy

from PIL import Image

# I would have loved to make this one a proper task, but time is limited and refactoring is slow.

base_path_association = "./data/association"
base_path_results = "./data/results"
path_output = "./data/boxplots"
association_list = [p for p in Path(base_path_association).glob("*") if (p/"association_full.csv").exists()]
Path(path_output).mkdir(parents=True, exist_ok=True)


max_seed = 17
threshold = 1e6
for threshold in [1.0e6, 2e6]:
    aggregated = []
    # aggregate all results into a single dataset
    for association_path in association_list:#[:1]:
        # experiment name
        name = association_path.name
        method = name.split("_")[0]
        model = [n for n in ["resnet18", "resnet34", "densenet121", "efficientnet_b0", "vgg16"] if n in name][0]
        dataset = [n for n in ["ABplus", "AB", "CO", "BigSmall", "colorGB", "isA", "leather", "metal_nut"] if n in name][0]
        if method=="ace":
            seed=int(name.split("_")[-1])
            training = name.split("_")[-2]
            variant="default"
        if (method=="eclad")or(method=="cshap"):
            seed=int(name.split("_")[-2])
            training = name.split("_")[-3]
            variant=name.split("_")[-1]
        if seed>max_seed:continue
        ####################### Load data
        # has results?
        if not (Path(base_path_results)/name/"results.json").exists(): 
            continue
        with open((Path(base_path_results)/name/"results.json").as_posix(), "rb") as f:
            results = json.load(f)
            if method=="ace": importance = [{"c":r["idx"], "name":r["name"], "importance":(r["score_mean"]*2-1), "concept_paths":r["concept_path"]} for r in results["results"]]
            if method=="eclad": importance = [{"c":r["idx"], "name":r["name"], "importance":r["RI"], "concept_paths":r["concept_paths"]} for r in results["results"]]
            if method=="cshap": importance = [{"c":r["idx"], "name":r["name"], "importance":r["shap_mean"], "concept_paths":r["concept_paths"]} for r in results["results"]]
        # has association?
        if not (Path(base_path_association)/name/"association.json").exists(): 
            continue
        with open((Path(base_path_association)/name/"association.json").as_posix(), "rb") as f:
            consolidated = json.load(f)
        ####################### compute correctness
        if not("name" in consolidated[0]):
            consolidated = [{**c, "name":f"c_{int(c['c']):02}"} for c in consolidated]
            continue
        consolidated = [{
            **c, 
            "importance":[i for i in importance if i["name"]==c["name"]][0]["importance"], 
            "concept_paths":[i for i in importance if i["name"]==c["name"]][0]["concept_paths"]} for c in consolidated]
        # get alignment
        dfc = pd.DataFrame(consolidated)
        important_primitives={
            "AB":[1,2],
            "ABplus":[1,2], 
            "CO":[1,2], 
            "BigSmall":[1,2], 
            "isA":[1,2], 
            "colorGB":[1], 
            "leather":[1,2,3,4,5,6], 
            "metal_nut":[1,2,3,4,5]
        }
        
        important_primitives=important_primitives[dataset]
        dfc["aligned"] = False
        dfc.loc[dfc["p"].isin(important_primitives)&(dfc["sym_dst"]<threshold), "aligned"] = True
        data = dfc.sort_values(by=["sym_dst"], ascending=True).drop_duplicates(["name"])
        ##### select aligned and unaligned concepts
        data = data[data["n_c"]!=0]
        representation_correctness = - data[(data["aligned"]==True)]["sym_dst"].abs().mean()
        importance_correctness_unaligned = data[(data["aligned"]==False)]["importance"].abs().mean()
        importance_correctness_align = data[(data["aligned"]==True)]["importance"].abs().mean()
        max_importance = data["importance"].abs().max()
        if max_importance==0: max_importance=1.0
        importance_correctness_diff = (importance_correctness_align - importance_correctness_unaligned)/max_importance
        n_concepts_aligned = data[(data["aligned"]==True)]["n_c"].count()
        n_concepts_unaligned = data[(data["aligned"]==False)]["n_c"].count()
        n_concepts = data["n_c"].count()
        ratio = data["ratio"].mean()
        ######### aggregate
        #print(name)
        model = {
            "resnet18":"r18", 
            "resnet34":"r34", 
            "densenet121":"den", 
            "efficientnet_b0":"eff", 
            "vgg16":"vgg"}[model]
        aggregated.append(
            {
                "name":name,
                "method":method,
                "model":model,
                "dataset":dataset,
                "training":training,
                "variant":variant,
                "seed":seed,
                "representation correctness":representation_correctness,
                "importance correctness unaligned":importance_correctness_unaligned,
                "importance correctness aligned":importance_correctness_align,
                "importance correctness":importance_correctness_diff,
                "n_concepts_aligned":n_concepts_aligned,
                "n_concepts_unaligned":n_concepts_unaligned,
                "n_concepts":n_concepts,
                "ratio":ratio
            }
        )
    dfa = pd.DataFrame(aggregated)
    dfa.to_csv((Path(path_output)/f"aggregated_results_t{threshold:.2E}.csv").as_posix(), sep=";")

    dfa = dfa[dfa["dataset"].isin(["AB", "BigSmall", "metal_nut", "leather"])]
    plt.rcParams.update({'font.size':40})
    for metric in [
                "representation correctness",
                "importance correctness"]:
        g = sns.catplot(x="model", y=metric,
                        hue="method", col="dataset",
                        data=dfa, kind="box",
                        height=8, aspect=1.4)
        fname = (Path(path_output)/f"boxplot_{metric}_t{threshold:.2E}_full.png")
        plt.savefig(fname, format="png", dpi=600, facecolor='white', bbox_inches='tight', transparent=False)
        plt.close()
