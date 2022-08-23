from pathlib import Path
import json
from PIL import Image
import numpy as np
import pandas as pd
import shutil
import pickle

from skimage import segmentation
import sklearn.cluster as cluster
from modules.utils.loggers import create_logger
from modules.activations.activations_pytorch import get_layers, get_gradients, get_activations
from .utils import patch_from_mask, encode_folder
from .tcav import tcav
import torch

def slice_slic(image, n_segments=[15,50,80], compactness=20.0, sigma=1.0, uniques=True):
    if not isinstance(n_segments, list): n_segments = [n_segments]
    masks=[]
    unique_masks = []
    # iterate over slic segmentations
    for n_seg in n_segments:
        # get segments
        segments = segmentation.slic(
            image, 
            n_segments=n_seg,
            compactness=compactness,
            sigma=sigma,
            start_label=0
        )
        if uniques:
            # check repeated segments
            for s in range(segments.max()):
                # verify that the segments are not repeated, as per ACE github implementation.
                mask = (segments == s).astype(float)
                unique = False
                if np.mean(mask) > 0.001:
                    unique = True
                    for seen_mask in unique_masks:
                        jaccard = np.sum(seen_mask * mask) / np.sum((seen_mask + mask) > 0)
                        if jaccard > 0.5:
                            unique = False
                            break
                if not unique:
                    segments[segments == s]=-1
        masks.append(segments.copy())
    # unique segments will have a mask index >0. non-unique segments will have a mask index of -1
    return masks

class ACE:
    def __init__(self, 
        save_path, 
        model,
        transforms,
        class_to_idx=None,
        class_paths=None,
        shape=(224,224),
        n_clusters = 25,
        n_segments = [15, 50, 80],
        sigma = 1.0,
        compactness = 20.0,
        filling=[125,125,125],
        n_limit_images = 100,

        tcav_layer=None,
        n_tcav_repetitions = 50,
        n_tcav_samples = 100,

        workers_activations=1,
        batch_size=8,
        workers_slic=1,
        workers_tcav=1,
        device="cpu",
        n_concept_examples=40,
        logger=None,
        rng=None,
        min_concept_samples=6
        ):
        """Initialization of the algorithm"""
        ##############################
        self.model = model
        self.transforms = transforms
        self.save_path = save_path
        self.class_to_idx = class_to_idx
        self.class_paths = class_paths
        self.shape = shape
        self.n_clusters = n_clusters
        self.n_segments = n_segments
        self.sigma = sigma
        self.compactness = compactness
        self.filling = filling
        self.n_limit_images = n_limit_images
        self.tcav_layer= tcav_layer
        self.n_tcav_repetitions = n_tcav_repetitions
        self.n_tcav_samples = n_tcav_samples
        self.workers_activations= workers_activations
        self.batch_size = batch_size
        self.workers_slic= workers_slic
        self.workers_tcav= workers_tcav
        self.device= device
        self.n_concept_examples= n_concept_examples
        self.min_concept_samples = min_concept_samples
        if logger is None: logger = create_logger("ACE")
        self.logger= logger
        if rng is None: self.rng = np.random.default_rng(0)
        elif isinstance(rng, int): self.rng = np.random.default_rng(rng)
        else: self.rng = rng
        self.classifiers = {}
        self.results = []

    def __call__(self):
        self.logger.debug("Starting ACE.")
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
        self.logger.debug("Finishing ACE.")

    def partial(self, class_name, class_idx):
        self.logger.debug(f"Starting ACE for class {class_name}.")
        # create result folder
        results_path = (Path(self.save_path)/class_name).as_posix()
        Path(results_path).mkdir(parents=True, exist_ok=True)
        # generate patches
        self.logger.debug(f"Generating patches.")
        self.generate_patches(self.class_paths[class_name], results_path)
        # encode patches
        self.logger.debug(f"Encoding patches.")
        acts_list, grads_list = self.encode_folder(results_path, class_idx)
        # cluster
        self.logger.debug(f"Clustering patches.")
        classifier, clusters = self.cluster(acts_list)
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
    
    def generate_patches(self, class_path, results_path):
        # glob images in folder
        image_paths = sorted([p for p in Path(class_path).glob("*") if p.suffix in [".jpg", ".jpeg", ".png", ".bmp", ".gif"]])[:self.n_limit_images]
        # iterate over images
        for p in image_paths:
            # open
            image = Image.open(p.as_posix()).convert('RGB')
            # slice
            masks = slice_slic(image, n_segments=self.n_segments, compactness=self.compactness, sigma=self.sigma, uniques=True)
            # for each mask
            for m_id, m in enumerate(masks):
                # for each patch id
                for i in range(int(m.max())):
                    if (m==i).sum()==0: continue
                    # get superpixel
                    superpixel, patch = patch_from_mask(image, (m==i))
                    # save superpixel
                    filename = Path(results_path)/(p.stem+f"_m_{m_id}_p_{i:03}.jpg")
                    superpixel = Image.fromarray(np.uint8(superpixel))
                    superpixel.save(filename, "JPEG")

    def encode_folder(self, results_path, class_idx):
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
        # cluster
        # cluster and filter
        km = cluster.KMeans(self.n_clusters)
        km.fit(acts_list)
        # assign to centers and get distance
        centers = km.cluster_centers_
        distances = np.linalg.norm(np.expand_dims(acts_list, 1) - np.expand_dims(centers, 0), ord=2, axis=-1)
        cluster_assignment, cost = np.argmin(distances, -1), np.min(distances, -1)
        # filter all clusters that are out of the percentile 95
        threshold = np.percentile(cost,95)
        # mark outliers
        cluster_assignment[cost<threshold]=-1
        # save clustering model
        return km, cluster_assignment
    
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
        r_paths = []
        for p in self.rng.choice(image_paths, min(n, len(image_paths)), replace=False):
            shutil.copy(p.as_posix(), (Path(results_path)/f"{p.parent.name}_{p.name}").as_posix())
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
        self.n_clusters = configs["n_clusters"]
        self.n_segments = configs["n_segments"]
        self.sigma = configs["sigma"]
        self.compactness = configs["compactness"]
        self.filling = configs["filling"]
        self.n_limit_images = configs["n_limit_images"]
        self.tcav_layer = configs["tcav_layer"]
        self.n_tcav_repetitions = configs["n_tcav_repetitions"]
        self.n_tcav_samples = configs["n_tcav_samples"]
        self.workers_activations = configs["workers_activations"]
        self.batch_size = configs["batch_size"]
        self.workers_slic = configs["workers_slic"]
        self.workers_tcav = configs["workers_tcav"]
        self.device = configs["device"]
        self.n_concept_examples = configs["n_concept_examples"]
        self.results = configs["results"]
        self.min_concept_samples = configs["min_concept_samples"]
        for class_name in self.class_to_idx.keys():
            with open((Path(self.save_path)/class_name/"classifier.pkl").as_posix(),"rb") as f:
                self.classifiers[class_name] = pickle.load(f)

    def save(self):
        config = {
            "class_to_idx": self.class_to_idx,
            "class_paths": self.class_paths,
            "shape": self.shape,
            "n_clusters" : self.n_clusters,
            "n_segments" : self.n_segments,
            "sigma" : self.sigma,
            "compactness" : self.compactness,
            "filling" : self.filling,
            "n_limit_images" : self.n_limit_images,
            "tcav_layer" : self.tcav_layer,
            "n_tcav_repetitions" : self.n_tcav_repetitions,
            "n_tcav_samples" : self.n_tcav_samples,
            "workers_activations" : self.workers_activations,
            "batch_size" : self.batch_size,
            "workers_slic" : self.workers_slic,
            "workers_tcav" : self.workers_tcav,
            "device" : self.device,
            "n_concept_examples" : self.n_concept_examples,
            "results": self.results,
            "min_concept_samples": self.min_concept_samples
        }
        with open((Path(self.save_path)/"results.json").as_posix(), 'w') as f:
            json.dump(config, f, indent=4)
        for class_name in self.class_to_idx.keys():
            with open((Path(self.save_path)/class_name/"classifier.pkl").as_posix(),"wb") as f:
                pickle.dump(self.classifiers[class_name], f)

    def mask(self, image, class_name):
        # load image
        if isinstance(image, str): image = Image.open(image).convert('RGB')
        image = image.resize(self.shape, Image.NEAREST)
        # slice image
        masks = slice_slic(image, n_segments=self.n_segments, compactness=self.compactness, sigma=self.sigma, uniques=False)
        # for each mask
        layers = get_layers(self.model, names=[self.tcav_layer])
        results = []
        for m_id, m in enumerate(masks):
            dummy_mask = m.copy()
            # for each patch id
            batch=[]
            batch_idx=[]
            for i in range(int(m.max())):
                if (m==i).sum()==0: continue
                # get superpixel
                superpixel, patch = patch_from_mask(image, (m==i))
                batch.append(superpixel)
                batch_idx.append(i)
                if (len(batch)<self.batch_size)and(i<int(m.max())-1):
                    continue
                else:
                    # get encoding
                    x = torch.cat([self.transforms(im).unsqueeze(0) for im in batch], 0)
                    outs, acts = get_activations(self.model, x.to(self.device), layers)
                    encodings = acts[self.tcav_layer].cpu().detach().numpy()
                    encodings = encodings.reshape(encodings.shape[0],-1)
                    # cluster
                    clusters = self.classifiers[class_name].predict(encodings)
                    for j in range(len(batch)):
                        dummy_mask[m==batch_idx[j]] = clusters[j]
                    batch=[]
                    batch_idx=[]
            results.append(dummy_mask)
        return results