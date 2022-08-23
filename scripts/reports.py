from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import json
from PIL import Image
from multiprocessing import Pool


def report(path_data="./data/results/eclad_AB_resnet18_plateau_0_n10s", path_output="./data/results_report", n=4, filters=[], size_multiplier=3):
    if not (Path(path_data)/"results.json").exists(): return
    path_dataset_config = list(Path(path_data).glob("dataset_*.json"))[0].as_posix()
    path_results = (Path(path_data)/"results.json").as_posix()
    Path(path_output).mkdir(parents=True, exist_ok=True)
    print(path_data)
    # load configs
    with open(Path(path_results).as_posix(), "r") as f:
        results = json.load(f)
    with open(Path(path_dataset_config).as_posix(), "r") as f:
        dataset_config = json.load(f)
    ##################
    if "n_segments" in results: 
        result_type = "ace"
    elif "n_slices" in results:
        result_type = "space"
    elif "eclad_layers" in results:
        result_type = "eclad"
    elif "cshap_layer" in results:
        result_type = "cshap"
    else:
        return 
    ################# get concepts
    if (result_type=="ace")or(result_type=="space"):
        concepts = [r for r in results["results"] if r["random_path"] is not None]
        if (len(filters)>0): concepts = [c for c in concepts if any([f==(Path(c["concept_path"]).parent.name) for f in filters])]
        concepts = sorted(concepts, key=lambda x: np.abs(x["score_mean"]), reverse=True)[:60]
    elif (result_type=="eclad"):
        concepts = results["results"]
        concepts = sorted(concepts, key=lambda x: np.abs(x["RI"]), reverse=True)[:60]
    elif (result_type=="cshap"):
        concepts = results["results"]
        concepts = sorted(concepts, key=lambda x: np.abs(x["shap_mean"]), reverse=True)[:60]

    if len(concepts)==0: return
    ################ plot concepts
    if (result_type=="ace")or(result_type=="space"):
        fig, axs = plt.subplots(len(concepts),n+1,figsize=(3*(n+1),len(concepts)*3))
        if len(concepts)==1: axs=axs.reshape(len(concepts),n+1)
        for idx in range(len(concepts)):
            r = concepts[idx]
            axs[idx, 0].bar(["tcav"],[r["score_mean"]])
            axs[idx, 0].set_ylabel(r["name"])
            axs[idx, 0].set_ylim(0, 1.1)
            axs[idx, 0].text(-0.38, 0.5, f'tcav:{r["score_mean"]:.2f}+-{r["score_std"]:.2f}', fontsize=15)
            axs[idx, 0].text(-0.38, 0.1, f'pval:{r["pval"]:.4f}', fontsize=15)
            ims = [p.as_posix() for p in Path(r["concept_path"]).glob("*") if p.suffix in [".jpg", ".jpeg", ".png", ".bmp", ".gif"]]
            for i in range(n):
                if i<len(ims)-1:
                    image = Image.open(ims[i]).convert('RGB')
                    axs[idx, i+1].imshow(image)
                else:
                    axs[idx, i+1].bar([""],[1],color=["grey"])
                axs[idx, i+1].set_axis_off()
    elif (result_type=="eclad"):
        name_classes = results["class_to_idx"].keys()
        n_info=1#4
        fig, axs = plt.subplots(len(concepts),(n*len(name_classes))+n_info,figsize=(size_multiplier*(n*len(name_classes)+n_info),len(concepts)*size_multiplier))
        if len(concepts)==1: axs=axs.reshape(len(concepts),(n*len(name_classes))+1)
        max_S = np.max(np.abs([r["ESk"] for r in concepts]))
        for idx in range(len(concepts)):
            r = concepts[idx]
            axs[idx, 0].bar(["RI"],[r["RI"]])
            axs[idx, 0].set_ylim(0, 1.1)
            axs[idx, 0].text(-0.38, 0.5, f'RI:{r["RI"]:.2f}', fontsize=15)

            #axs[idx, 1].bar(["RIc"],[r["RIc"]])
            #axs[idx, 1].set_ylim(0, 1.1)
            #axs[idx, 1].text(-0.38, 0.5, f'RIc:{r["RIc"]:.2f}', fontsize=15)
            # add ratios

            #axs[idx, 2].set_ylabel(r["name"])
            #axs[idx, 2].scatter(name_classes,r["ESk"])
            #axs[idx, 2].axhline(y=0, color='gray', linestyle='-')
            #axs[idx, 2].set_ylim(-1.1*np.max(np.abs(r["ESk"])), 1.1*np.max(np.abs(r["ESk"])))
            #axs[idx, 2].set_yscale('symlog')
            #axs[idx, 2].grid(visible=True, which='both')

            #axs[idx, 3].set_ylabel(r["name"])
            #axs[idx, 3].scatter(name_classes,r["ratios"])
            #axs[idx, 3].axhline(y=0, color='gray', linestyle='-')
            #axs[idx, 3].set_ylim(-0.1, 1.1)
            #axs[idx, 3].grid(visible=True, which='both')
            for class_name, class_idx in results["class_to_idx"].items():
                ims = [p.as_posix() for p in Path(r["concept_paths"][class_name]).glob("*") if p.suffix in [".jpg", ".jpeg", ".png", ".bmp", ".gif"]]
                for i in range(n):
                    if i<len(ims)-1:
                        image = Image.open(ims[i]).convert('RGB')
                        axs[idx, i+n_info+class_idx*n].imshow(image)
                    else:
                        axs[idx, i+n_info+class_idx*n].bar([""],[1],color=["grey"])
                    axs[idx, i+n_info+class_idx*n].set_axis_off()
                    axs[idx, i+n_info+class_idx*n].set_title(class_name)
    elif (result_type=="cshap"):
        name_classes = results["class_to_idx"].keys()
        fig, axs = plt.subplots(len(concepts),(n*len(name_classes))+1,figsize=(size_multiplier*(n*len(name_classes)+1),len(concepts)*size_multiplier))
        if len(concepts)==1: axs=axs.reshape(len(concepts),(n*len(name_classes))+1)
        for idx in range(len(concepts)):
            r = concepts[idx]
            axs[idx, 0].bar(["shap"],[r["shap_mean"]])
            axs[idx, 0].set_ylim(0, 1.1)
            axs[idx, 0].text(-0.38, 0.5, f'shap:{r["shap_mean"]:.2f}', fontsize=15)
            axs[idx, 0].grid(visible=True, which='both')
            for class_name, class_idx in results["class_to_idx"].items():
                ims = [p.as_posix() for p in Path(r["concept_paths"][class_name]).glob("*") if p.suffix in [".jpg", ".jpeg", ".png", ".bmp", ".gif"]]
                for i in range(n):
                    if i<len(ims)-1:
                        image = Image.open(ims[i]).convert('RGB')
                        axs[idx, i+1+class_idx*n].imshow(image)
                    else:
                        axs[idx, i+1+class_idx*n].bar([""],[1],color=["grey"])
                    axs[idx, i+1+class_idx*n].set_axis_off()
                    axs[idx, i+1+class_idx*n].set_title(class_name)
    result_name = (Path(path_output)/(Path(path_data).name+".png")).as_posix()
    fig.savefig(result_name)
    plt.close()

def report_wrapper(kwargs):
    return report(**kwargs)

def reports(path_data="./data/results", path_output="./data/results_report", n=5, class_filters=[], folder_filters=[], workers=1, size_multiplier=3):
    pool = Pool(workers)
    kwargs_list=[dict(path_data=p.as_posix(), path_output=path_output, n=n, filters=class_filters,  size_multiplier=size_multiplier) for p in Path(path_data).glob("*") if any([s in p.name for s in folder_filters]+[len(folder_filters)==0])]
    pool_results_batch = pool.map(report_wrapper,kwargs_list)
    pool.close()
    pool.join()
    
if __name__ == '__main__':
    reports(class_filters=[], folder_filters=[], workers=1, size_multiplier=3)