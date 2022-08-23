from pathlib import Path
import json
from PIL import Image
import numpy as np
import pandas as pd
import shutil
import pickle
import copy

import sklearn.cluster as cluster
from sklearn import preprocessing
from sklearn.decomposition import PCA
from modules.utils.loggers import create_logger
from modules.activations.activations_pytorch import get_layers, get_gradients, get_activations
from .utils import patch_from_mask, encode_folder
from .tcav import tcav
import torch
from torch.nn import functional as F
import torchvision
from captum.attr import GuidedGradCam

def slice_importance_grid(image, model, transforms, gradcam_layer, out_idx, n_slices=[6], n_percentage=0.1, device="cpu"):
    if not isinstance(n_slices, list): n_slices = [n_slices]
    masks=[]
    # transform image
    x = transforms(image)
    shape = x.numpy().shape[-2:]
    x = x.to(device)
    # compute gradcam
    model.to(device)
    layers = get_layers(model, names=[gradcam_layer])
    guided_gc = GuidedGradCam(model, layers[gradcam_layer])
    x.requires_grad=True
    #print(device, x.device, next(model.parameters()).device)
    attribution = guided_gc.attribute(x.unsqueeze(0), out_idx)[0]
    attribution = F.relu(attribution).mean(0).detach().cpu().numpy()
    # compute and filter mask
    for n_slice in n_slices:
        # create mask
        mask = - np.ones(shape, dtype=int)
        for c in range(n_slice):
            for r in range(n_slice):
                mask[int(r*shape[0]/n_slice):int((r+1)*shape[0]/n_slice),int(c*shape[0]/n_slice):int((c+1)*shape[0]/n_slice)]= int(c + r*n_slice)
        # compute aggregated importance of each mask
        agg_importance=[]
        for mask_id in range(mask.max()+1):
            attribution_positives = (((mask==mask_id)*attribution)>0).sum()
            attribution_sum = ((mask==mask_id)*attribution).sum()
            agg_importance.append(0 if attribution_positives==0 else attribution_sum/attribution_positives)
        # select masks given ranking
        ranking = np.argsort(agg_importance)[::-1]
        ranking = ranking[:int(max(len(agg_importance)*n_percentage,1))] #get the top percentage
        for mask_id in range(mask.max()+1):
            if not (mask_id in ranking):
                mask[mask==mask_id]=-1
        # add mask
        masks.append(mask.copy())
    # return masks
    return masks

class Tile(object):
    def __init__(self, n_slice):
        self.n_slice = n_slice
    def __call__(self, img):
        new_im = Image.new('RGB', (img.size[0]*self.n_slice, img.size[1]*self.n_slice))
        for i in range(self.n_slice):
            for j in range(self.n_slice):
                new_im.paste(img, (img.size[0]*i, img.size[1]*j))
        return new_im

    def __repr__(self):
        return self.__class__.__name__ + '('+str(self.n_slice)+')'

class SPACE:
    def __init__(self, 
        save_path, 
        model,
        transforms,
        class_to_idx,
        class_paths,
        shape=(224,224),

        n_slices=6, 
        n_percentage=0.1,
        dim_pca = 25,
        gradcam_layer = None,
        n_limit_images = 100,

        tcav_layer=None,
        n_tcav_repetitions = 50,
        n_tcav_samples = 100,

        workers_activations=1,
        batch_size=8,
        workers_tcav=1,
        device="cpu",
        n_concept_examples=40,
        raw_random=False,
        logger=None,
        rng=None,
        min_concept_samples=10
        ):
        """Initialization of the algorithm"""
        ##############################
        self.model = model
        self.transforms = transforms
        self.save_path = save_path
        self.class_to_idx = class_to_idx
        self.class_paths = class_paths
        self.shape = shape
        self.n_slices = n_slices
        self.n_percentage = n_percentage
        self.dim_pca = dim_pca
        self.gradcam_layer = gradcam_layer
        self.n_limit_images = n_limit_images
        self.tcav_layer= tcav_layer
        self.n_tcav_repetitions = n_tcav_repetitions
        self.n_tcav_samples = n_tcav_samples
        self.workers_activations= workers_activations
        self.batch_size = batch_size
        self.workers_tcav= workers_tcav
        self.device= device
        self.n_concept_examples= n_concept_examples
        self.raw_random = raw_random
        self.min_concept_samples = min_concept_samples
        if logger is None: logger = create_logger("SPACE")
        self.logger= logger
        if rng is None: self.rng = np.random.default_rng(0)
        elif isinstance(rng, int): self.rng = np.random.default_rng(rng)
        else: self.rng = rng
        self.classifiers = {}
        self.results = []

    def __call__(self):
        self.logger.debug("Starting SPACE.")
        # create result folder
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        # iterate over classes
        self.results = []
        for class_name, class_idx in self.class_to_idx.items():
            # analyze class
            class_results = self.partial(class_name, class_idx)
            self.results.extend(class_results)
        # save results
        self.save()
        self.clean()
        self.logger.debug("Finishing SPACE.")

    def partial(self, class_name, class_idx):
        self.logger.debug(f"Starting SPACE for class {class_name}.")
        # create result folder
        results_path = (Path(self.save_path)/class_name).as_posix()
        Path(results_path).mkdir(parents=True, exist_ok=True)
        # generate patches
        self.logger.debug(f"Generating patches.")
        self.generate_patches(self.class_paths[class_name], results_path, class_idx)
        # encode patches
        self.logger.debug(f"Encoding patches.")
        acts_list, grads_list = self.encode_folder(results_path, class_idx)
        # cluster
        self.logger.debug(f"Clustering patches.")
        classifier, clusters = self.cluster(acts_list)
        self.n_clusters = clusters.max()+1
        self.classifiers[class_name] = classifier
        with open((Path(results_path)/"classifier.pkl").as_posix(),"wb") as f:
            pickle.dump(classifier, f)
        # move
        self.logger.debug(f"Moving patches.")
        concept_paths = self.move(results_path, clusters, acts_list, grads_list, class_name, class_idx)
        # for each concept
        results = []
        self.logger.debug(f"Testing concepts.")
        for idx in range(self.n_clusters):
            available_images = len([p for p in Path(concept_paths[idx]).glob("*") if p.suffix in [".jpg", ".jpeg", ".png", ".bmp", ".gif"]])
            if available_images>=self.min_concept_samples:
                # build random concept
                random_path = (Path(results_path)/f"r_{idx:02}_{class_name}").as_posix()
                self.build_random(self.class_paths, random_path, n=self.n_tcav_samples*5, save_list=True)
                # test concept
                cav, score_mean, score_std, pval, acc, available_images = self.tcav(
                    concept_paths[idx], 
                    random_path,
                    self.class_paths[class_name],
                    results_path,
                    class_idx=class_idx,
                    n=self.n_tcav_repetitions, 
                    workers=self.workers_tcav)
            else:
                cav, score_mean, score_std, pval, acc = None, 0.5, 0, 1.0, 0.0
                random_path = None
            results.append(
                {
                    "cav":cav, "score_mean":score_mean, "score_std":score_std, "pval":pval, "acc":acc, 
                    "concept_path":concept_paths[idx], "random_path":random_path, 
                    "idx":idx, "name":f"c_{idx:02}_{class_name}", "n":available_images, "class":class_name
                }
            )
            self.logger.debug(f"Concept c_{idx:02}_{class_name}: score:{score_mean:.3f}+-{score_std:.3f} pval:{pval:.3f} acc:{acc:.3f} n:{available_images}")
        # save results
        filename = (Path(self.save_path)/class_name/"results.json").as_posix()
        with open(filename, 'w') as f:
            json.dump(results, f, indent=4)
        return results
    
    def generate_patches(self, class_path, results_path, class_idx):
        # glob images in folder
        image_paths = sorted([p for p in Path(class_path).glob("*") if p.suffix in [".jpg", ".jpeg", ".png", ".bmp", ".gif"]])[:self.n_limit_images]
        # iterate over images
        for p in image_paths:
            # open
            image = Image.open(p.as_posix()).convert('RGB')
            image = image.resize(self.shape)
            # slice
            self.model.to(self.device)
            masks = slice_importance_grid(image, self.model, self.transforms, self.gradcam_layer, class_idx, n_slices=[self.n_slices], n_percentage=self.n_percentage, device=self.device)
            self.model.to(self.device)
            # for each mask
            for m_id, m in enumerate(masks):
                # for each patch id
                for i in range(int(m.max())+1):
                    if (m==i).sum()==0: continue
                    # get superpixel
                    superpixel, patch = patch_from_mask(image, (m==i))
                    # save superpixel
                    filename = Path(results_path)/(p.stem+f"_m_{m_id}_p_{i:03}.jpg")
                    superpixel = Image.fromarray(np.uint8(superpixel))
                    superpixel = Tile(self.n_slices)(superpixel)
                    superpixel.save(filename, "JPEG")

    def encode_folder(self, results_path, class_idx):
        #transforms_tiled = copy.copy(self.transforms)
        #transforms_tiled.transforms = [Tile(self.n_slices)]+transforms_tiled.transforms
        acts_list, grads_list = encode_folder(
            self.model, 
            results_path,
            results_path,
            self.tcav_layer, class_idx, 
            self.transforms, 
            workers=self.workers_activations, 
            batch_size=self.batch_size, 
            device=self.device)
        return acts_list, grads_list
    
    def cluster(self, acts_list):
        # reshape
        acts_list = acts_list.reshape(acts_list.shape[0],-1)
        # standardize
        acts_list = preprocessing.scale(acts_list)
        # PCA
        if self.dim_pca is not None:
            pca = PCA(n_components=self.dim_pca)
            acts_list = pca.fit_transform(acts_list)
        # cluster 
        clustering = cluster.OPTICS(metric="manhattan").fit(acts_list)
        # assignment
        cluster_assignment = clustering.labels_
        # save clustering model
        return clustering, cluster_assignment
    
    def move(self, results_path, clusters, acts_list, grads_list, class_name, class_idx):
        # get list of images
        image_paths = sorted([p for p in Path(results_path).glob("*") if p.suffix in [".jpg", ".jpeg", ".png", ".bmp", ".gif"]])
        # iterate over clusters
        concept_paths = {}
        for idx in range(-1, self.n_clusters):
            if idx==-1: concept_path = (Path(results_path)/f"outliers").as_posix()
            else: concept_path = (Path(results_path)/f"c_{idx:02}_{class_name}").as_posix()
            Path(concept_path).mkdir(parents=True, exist_ok=True)
            concept_paths[idx] = concept_path
            # move patches
            concept_patches = [image_paths[i] for i in range(len(clusters)) if clusters[i]==idx]
            for img_path in concept_patches:
                shutil.move(img_path, (Path(concept_path)/Path(img_path).name).as_posix())
            # move activations and gradients
            np.save((Path(concept_path)/f"acts_c_{idx:02}_{class_name}.npy").as_posix(), acts_list[clusters==idx,:])
            np.save((Path(concept_path)/f"grads_c_{idx:02}_{class_name}_{class_idx}.npy").as_posix(), grads_list[clusters==idx,:])
        (Path(results_path)/f"acts.npy").unlink(missing_ok=False)
        (Path(results_path)/f"grads_{class_idx}.npy").unlink(missing_ok=False)
        return concept_paths

    def build_random(self, paths_input, results_path, n=100, save_list=False):
        Path(results_path).mkdir(parents=True, exist_ok=True)
        # get list of images
        image_paths = sorted([p for class_path in paths_input.values() for p in Path(class_path).glob("*") if p.suffix in [".jpg", ".jpeg", ".png", ".bmp", ".gif"]])
        cropper = torchvision.transforms.RandomCrop(self.shape)
        r_paths = []
        for p in self.rng.choice(image_paths, min(n, len(image_paths)), replace=False):
            if self.raw_random:
                shutil.copy(p.as_posix(), (Path(results_path)/f"{p.parent.name}_{p.name}").as_posix())
            else: 
                # get random patch
                image = Image.open(p.as_posix()).convert('RGB')
                superpixel = cropper(image)
                # save patch
                filename = Path(results_path)/(f"{p.parent.name}_{p.stem}_p_001.jpg")
                superpixel.save(filename, "JPEG")
            r_paths.append(p.as_posix())
        # save list of paths composing this random
        if save_list:
            with open(results_path+".json", "w") as f:
                json.dump(r_paths, f, indent=4)

    def tcav(self, concept_path, random_path, class_path, results_path, class_idx, n=50, workers=1):
        cav, score_mean, score_std, pval, acc, available_images = tcav(
            self.model, self.transforms, self.tcav_layer, 
            concept_path, random_path, 
            class_path, results_path, 
            class_idx, samples=self.n_tcav_samples, n=self.n_tcav_repetitions, 
            device=self.device, batch_size=self.batch_size, rng=self.rng)
        return cav, score_mean, score_std, pval, acc, available_images
    
    def clean(self):
        for class_name, class_idx in self.class_to_idx.items():
            for c_p in (Path(self.save_path)/class_name).glob("*"):
                if c_p.is_dir():
                    image_paths = sorted([p for p in Path(c_p).glob("*") if p.suffix in [".jpg", ".jpeg", ".png", ".bmp", ".gif"]])
                    for p in image_paths[self.n_concept_examples:]:
                        p.unlink()
        for p in Path(self.save_path).glob("**/*.npy"):
            p.unlink()

    def load(self):
        with open((Path(self.save_path)/"results.json").as_posix(), 'r') as f:
            configs=json.load(f)
        self.class_to_idx = configs["class_to_idx"]
        self.class_paths = configs["class_paths"]
        self.shape = configs["shape"]
        self.n_slices = configs["n_slices"]
        self.n_percentage = configs["n_percentage"]
        self.dim_pca = configs["dim_pca"]
        self.n_limit_images = configs["n_limit_images"]
        self.tcav_layer = configs["tcav_layer"]
        self.n_tcav_repetitions = configs["n_tcav_repetitions"]
        self.n_tcav_samples = configs["n_tcav_samples"]
        self.workers_activations = configs["workers_activations"]
        self.batch_size = configs["batch_size"]
        self.workers_tcav = configs["workers_tcav"]
        self.device = configs["device"]
        self.n_concept_examples = configs["n_concept_examples"]
        self.raw_random = configs["raw_random"]
        self.min_concept_samples = configs["min_concept_samples"]
        self.results = configs["results"] 
        for class_name in self.class_to_idx.keys():
            with open((Path(self.save_path)/class_name/"classifier.pkl").as_posix(),"rb") as f:
                self.classifier[class_name] = pickle.load(f)

    def save(self):
        config = {
            "class_to_idx": self.class_to_idx,
            "class_paths": self.class_paths,
            "shape": self.shape,
            "n_slices" : self.n_slices,
            "n_percentage" : self.n_percentage,
            "dim_pca" : self.dim_pca,
            "gradcam_layer" : self.gradcam_layer,
            "n_limit_images" : self.n_limit_images,
            "tcav_layer" : self.tcav_layer,
            "n_tcav_repetitions" : self.n_tcav_repetitions,
            "n_tcav_samples" : self.n_tcav_samples,
            "workers_activations" : self.workers_activations,
            "batch_size" : self.batch_size,
            "workers_tcav" : self.workers_tcav,
            "device" : self.device,
            "n_concept_examples" : self.n_concept_examples,
            "raw_random": self.raw_random,
            "min_concept_samples" : self.min_concept_samples,
            "results": self.results
        }
        with open((Path(self.save_path)/"results.json").as_posix(), 'w') as f:
            json.dump(config, f, indent=4)
        for class_name in self.class_to_idx.keys():
            with open((Path(self.save_path)/class_name/"classifier.pkl").as_posix(),"wb") as f:
                pickle.dump(self.classifiers[class_name], f)
