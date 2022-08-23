from pathlib import Path
import json
from PIL import Image
import numpy as np
import pandas as pd
import shutil
import pickle
import gc

import torch
from PIL import Image
from torch.nn import functional as f
from sklearn.cluster import Birch, MiniBatchKMeans, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from torchvision import transforms

from modules.utils.loggers import create_logger
from modules.activations.activations_pytorch import get_layers, get_gradients, get_activations
from modules.xai.utils import DummyDataset
class ECLAD:
    def __init__(self, 
        save_path, 
        model,
        transforms,
        class_to_idx=None,
        class_paths=None,
        shape=(224,224),
        n_limit_images = 100,

        n_clusters = 10,
        eclad_layers=None,
        upsampling_mode="bilinear",
        clustering_method="MiniBatchKMeans",
        eclad_clustering_batches=2,
        eclad_sample=1.0,
        eclad_alpha=0.5,
        eclad_standardscale = False,

        workers_activations=1,
        batch_size=8,
        device="cpu",
        n_concept_examples=40,
        logger=None,
        rng=None
        ):
        """Initialization of the algorithm"""
        ##############################
        self.save_path = save_path
        self.model = model
        self.transforms = transforms
        self.class_to_idx = class_to_idx
        self.class_paths = class_paths
        self.shape = shape
        self.n_limit_images = n_limit_images
        self.n_clusters = n_clusters
        self.eclad_layers = eclad_layers
        self.upsampling_mode = upsampling_mode 
        self.clustering_method = clustering_method
        self.eclad_clustering_batches = eclad_clustering_batches
        self.eclad_sample = eclad_sample
        self.eclad_alpha = eclad_alpha
        self.eclad_standardscale = eclad_standardscale
        self.workers_activations = workers_activations
        self.batch_size = batch_size
        self.device = device
        self.n_concept_examples = n_concept_examples

        if logger is None: logger = create_logger("ECLAD")
        self.logger= logger
        if rng is None: self.rng = np.random.default_rng(0)
        elif isinstance(rng, int): self.rng = np.random.default_rng(rng)
        else: self.rng = rng
        self.classifier = None
        self.scaler = None
        self.centroids = None
        self.results = []

    def __call__(self):
        self.logger.debug("Starting ECLAD.")
        # create result folder
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        # extract concepts
        self.logger.debug("ECLAD: Clustering started.")
        shuffled_images = [p.as_posix() for class_path in self.class_paths.values() for p in sorted(list(Path(class_path).glob("*")))[:self.n_limit_images] if p.suffix in [".jpg", ".jpeg", ".png", ".bmp", ".gif"]]
        self.rng.shuffle(shuffled_images)
        self.cluster_descriptors(
            save_path=self.save_path, 
            images_paths=shuffled_images
        )
        self.logger.debug("ECLAD: Clustering finished.")
        # extract examples
        self.logger.debug("ECLAD: Generating examples started.")
        for class_name, class_idx in self.class_to_idx.items():
            # create folder for examples
            class_path = (Path(self.save_path)/class_name).as_posix()
            Path(class_path).mkdir(parents=True, exist_ok=True)
            # generate examples for images of this class
            class_images = [p.as_posix() for p in Path(self.class_paths[class_name]).glob("*") if p.suffix in [".jpg", ".jpeg", ".png", ".bmp", ".gif"]]
            self.generate_examples(class_images[:self.n_limit_images], save_path = class_path, n = self.n_concept_examples)
        self.logger.debug("ECLAD: Generating examples finished.")
        # test concepts
        self.logger.debug("ECLAD: Testing concepts started.")
        self.evaluate_importance()
        self.logger.debug("ECLAD: Testing concepts finished.")
        # save results
        self.save()
        self.clean()
        self.logger.debug("Finishing ECLAD.")

    def cluster_descriptors(self, save_path=None, images_paths=None):
        if not (Path(save_path)/"classifier.pkl").exists():
            layers = get_layers(self.model, names=self.eclad_layers)
            # create loaders
            ds = DummyDataset(images_paths, self.transforms)
            ds_loader = torch.utils.data.DataLoader(
                ds, 
                batch_size=self.batch_size, 
                num_workers=self.workers_activations, 
                shuffle=False)
            
            # create scaler and clustering model
            sc= StandardScaler()
            if self.clustering_method == "KMeans": 
                clst = KMeans(n_clusters=self.n_clusters, random_state=0)
            elif self.clustering_method == "MiniBatchKMeans": 
                clst = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=0, batch_size=int(self.shape[0]*self.shape[1]*self.eclad_sample)*self.eclad_clustering_batches)
            elif (self.clustering_method == "Birch") or (self.clustering_method == "MinibatchBirch"): 
                clst = Birch(branching_factor=20, n_clusters=self.n_clusters, compute_labels=True,copy=False)
            elif self.clustering_method == "GaussianMixture": 
                clst = GaussianMixture(n_components=self.n_clusters, random_state=0)
            else: 
                clst = KMeans(n_clusters=self.n_clusters, random_state=0)
            
            if (self.clustering_method == "MiniBatchKMeans") or (self.clustering_method == "MinibatchBirch"):
                # if we want to do it by batches
                minibatch_LADs = []
                if self.eclad_standardscale:
                    for xb,yb in ds_loader:
                        # Get activations
                        LADs = self.get_lads(xb).detach().cpu().numpy()
                        if self.eclad_sample<=1.0: LADs = self.rng.choice(LADs,size=int(self.shape[0]*self.shape[1]*self.eclad_sample),replace=False)
                        minibatch_LADs.append(LADs)
                        # if we have a full batch
                        if len(minibatch_LADs)>= self.eclad_clustering_batches:
                            # flatten batch
                            minibatch_LADs = np.concatenate(minibatch_LADs, axis=0)
                            # partial fit scaler
                            sc.partial_fit(minibatch_LADs)
                            # clean batch
                            del LADs, minibatch_LADs
                            minibatch_LADs = []
                            gc.collect()
                            torch.cuda.empty_cache()
                
                for xb,yb in ds_loader:
                    # Get activations
                    LADs = self.get_lads(xb).detach().cpu().numpy()
                    if self.eclad_sample<=1.0: LADs = self.rng.choice(LADs,size=int(self.shape[0]*self.shape[1]*self.eclad_sample),replace=False)
                    minibatch_LADs.append(LADs)
                    # if we have a full batch
                    if len(minibatch_LADs)>= self.eclad_clustering_batches:
                        # flatten batch
                        minibatch_LADs = np.concatenate(minibatch_LADs, axis=0)
                        # partial fit clustering
                        if self.eclad_standardscale:
                            clst.partial_fit(sc.transform(minibatch_LADs))
                        else:
                            clst.partial_fit(minibatch_LADs)
                        # clean batch
                        del LADs, minibatch_LADs
                        minibatch_LADs = []
                        gc.collect()
                        torch.cuda.empty_cache()
            else:
                # if we don't want to do it by batches:
                minibatch_LADs = []
                for xb,yb in ds_loader:
                    # Get activations
                    LADs = self.get_lads(xb).detach().cpu().numpy()
                    if self.eclad_sample<=1.0: LADs = self.rng.choice(LADs,size=int(self.shape[0]*self.shape[1]*self.eclad_sample),replace=False)
                    minibatch_LADs.append(LADs)
                minibatch_LADs = np.concatenate(minibatch_LADs, axis=0)
                if self.eclad_standardscale:
                    clst.fit(sc.fit_transform(minibatch_LADs))
                else:
                    clst.fit(minibatch_LADs)
                # clean vars
                del minibatch_LADs, LADs
                gc.collect()
                torch.cuda.empty_cache()
            with open((Path(save_path)/"scaler.pkl").as_posix(),"wb") as f:
                pickle.dump(sc, f)
            with open((Path(save_path)/"classifier.pkl").as_posix(),"wb") as f:
                pickle.dump(clst, f)
        else:
            with open((Path(save_path)/"scaler.pkl").as_posix(),"rb") as f:
                sc = pickle.load(f)
            with open((Path(save_path)/"classifier.pkl").as_posix(),"rb") as f:
                clst = pickle.load(f)
        # get centroids
        if "means_" in clst.__dict__: 
            centroids = clst.means_
        else:
            centroids = clst.cluster_centers_
        self.scaler = sc
        self.classifier = clst
        self.centroids = centroids.tolist()
    
    def get_lads(self, x, model=None, layers=None, size=None, sample=None, mode=None, device=None):
        if model is None: model = self.model
        if layers is None: layers = get_layers(model, names=self.eclad_layers)
        if size is None: size = self.shape
        if sample is None: sample = self.eclad_sample
        if mode is None: mode = self.upsampling_mode
        if device is None: device = self.device
        # evaluate x
        outs, acts = get_activations(model, x.to(self.device), layers)
        # interpolate layers
        if mode!="nearest": activations = [f.interpolate(a, size=size, mode=mode, align_corners=True) for a in acts.values()]
        else: activations = [f.interpolate(a, size=size, mode=mode) for a in acts.values()]
        # aggregated through depth
        descriptors = torch.cat(activations, dim=1)
        # reshape to loose other dimensions (batch_size, channels, height, width)
        descriptors = descriptors.permute(0,2,3,1).flatten(0,2)
        return descriptors

    def generate_examples(self, image_paths, save_path, n=10):
        Path(save_path).mkdir(parents=True, exist_ok=True)
        counters = [n for i in range(self.n_clusters)]
        for img_path in image_paths:
            # open image
            image = Image.open(img_path).convert('RGB')
            # mask
            mask = self.mask(image)
            image = image.resize(self.shape)
            mask = mask[..., np.newaxis]
            # save examples
            for concept_idx in range(self.n_clusters):
                (Path(save_path)/f"c_{concept_idx:02}").mkdir(parents=True, exist_ok=True)
                if (counters[concept_idx]>0) and (np.sum(mask==concept_idx)>0):
                    # save masked image
                    masked_image = (np.array(image)*(mask==concept_idx) + np.array(image)*(mask!=concept_idx)*self.eclad_alpha).astype(np.uint8)
                    masked_image = Image.fromarray(masked_image)
                    save_name = (Path(save_path)/f"c_{concept_idx:02}"/Path(img_path).name).as_posix()
                    masked_image.save(save_name)
                    # update counters
                    counters[concept_idx] = counters[concept_idx]-1
            # check if all examples were found
            if sum(counters)==0:
                break

    def evaluate_importance(self):
        # get layers
        layers = get_layers(self.model, names=self.eclad_layers)
        # init list with sensitivities
        eval_list = []
        # iterate over classes to test (this defines how the grad is computed)
        for class_name, class_idx in self.class_to_idx.items():
            # iterate over images from each class (this defines the images to use)
            for class_name_test, class_idx_test in self.class_to_idx.items():
                # get limited list of images per class
                class_images = [p.as_posix() for p in Path(self.class_paths[class_name_test]).glob("*") if p.suffix in [".jpg", ".jpeg", ".png", ".bmp", ".gif"]]
                # create loaders
                ds = DummyDataset(class_images[:self.n_limit_images], self.transforms)
                ds_loader = torch.utils.data.DataLoader(
                    ds,
                    batch_size=self.batch_size,
                    num_workers=self.workers_activations,
                    shuffle=False)
                # iterate over all images in the class
                img_idx = 0
                for xb,yb in ds_loader:
                    # evaluate x
                    outs, acts, grads = get_gradients(self.model, xb.to(self.device), layers, out_idx=class_idx)
                    # interpolate layers
                    if self.upsampling_mode!="nearest": 
                        activations = [f.interpolate(a, size=self.shape, mode=self.upsampling_mode, align_corners=True) for a in acts.values()]
                        gradients = [f.interpolate(g, size=self.shape, mode=self.upsampling_mode, align_corners=True) for g in grads.values()]
                    else: 
                        activations = [f.interpolate(a, size=self.shape, mode=self.upsampling_mode) for a in acts.values()]
                        gradients = [f.interpolate(g, size=self.shape, mode=self.upsampling_mode) for g in grads.values()]
                    # aggregated through depth and reshape
                    LADs = torch.cat(activations, dim=1).permute(0,2,3,1).flatten(0,2).detach()
                    LADs_gradient = torch.cat(gradients, dim=1).permute(0,2,3,1).flatten(0,2).detach()
                    # find concepts
                    if self.eclad_standardscale:
                        labels = self.classifier.predict(self.scaler.transform(LADs.cpu().numpy()))
                    else:
                        labels = self.classifier.predict(LADs.cpu().numpy())
                    # pixel wise sensitivity
                    LADs_sensitivity = torch.sum(LADs_gradient * LADs, axis=1).cpu().numpy()
                    # for each image
                    for idx_sample in range(xb.shape[0]):
                        img_sensitivity = LADs_sensitivity[idx_sample*(self.shape[0]*self.shape[1]):(idx_sample+1)*(self.shape[0]*self.shape[1])]
                        im_concepts = labels[idx_sample*(self.shape[0]*self.shape[1]):(idx_sample+1)*(self.shape[0]*self.shape[1])]
                        # for each concept
                        for c in range(self.n_clusters):
                            # aggregate to evaluation
                            c_idxs = (im_concepts==c)
                            n = np.sum(c_idxs)
                            c_sensitivities = img_sensitivity[c_idxs]
                            eval_list.append(
                                {
                                    "image_path":class_images[img_idx],
                                    "image_idx":img_idx,
                                    "y":class_idx_test,
                                    "y_grad":class_idx,
                                    "c":c,
                                    "n":n,
                                    "s_mean":0 if n==0 else np.mean(c_sensitivities),
                                    "s_std":0 if n==0 else np.std(c_sensitivities),
                                }
                            )
                        img_idx+=1
        df = pd.DataFrame(eval_list)
        df.to_csv((Path(self.save_path)/"evaluation_extended.csv").as_posix(), sep=";")
        # compute importances
        results = [{
            "centroid" : self.centroids[i], 
            "idx":i, 
            "name":f"c_{i:02}", 
            "concept_paths":{k:(Path(self.save_path)/k/f"c_{i:02}").as_posix() for k in self.class_to_idx.keys()},
            "CS":[0 for k in self.class_to_idx.keys()],
            "ESk":[0 for k in self.class_to_idx.keys()],
            "ESkn":[0 for k in self.class_to_idx.keys()],
            "RI":0,
            "RIc":0,
            "ratios":[0 for k in self.class_to_idx.keys()]
            } for i in range(self.n_clusters)]

        # compute contrastive sensitivity:
        for c in range(self.n_clusters):
            for k_class in range(len(self.class_to_idx.values())):
                # use grads sensitivity for this class
                idx_k_c = (df["y"]==k_class)&(df["c"]==c)&(df["y_grad"]==k_class)
                idx_kn_c = (df["y"]!=k_class)&(df["c"]==c)&(df["y_grad"]==k_class)
                E_S_k_c = np.sum(df[idx_k_c]["n"]*df[idx_k_c]["s_mean"])/np.sum(df[idx_k_c]["n"]) if np.sum(df[idx_k_c]["n"])>0 else 0
                E_S_kn_c = np.sum(df[idx_kn_c]["n"]*df[idx_kn_c]["s_mean"])/np.sum(df[idx_kn_c]["n"]) if np.sum(df[idx_kn_c]["n"])>0 else 0
                CS_k_c = E_S_k_c - E_S_kn_c
                results[c]["CS"][k_class] = CS_k_c
                results[c]["ESk"][k_class] = E_S_k_c
                results[c]["ESkn"][k_class] = E_S_kn_c
                results[c]["ratios"][k_class] = len(df[(df["c"]==c)&(df["n"]>0)&(df["y"]==k_class)]["image_path"].unique())/len(df[df["y"]==k_class]["image_path"].unique())
        # compute relative importance
        for c in range(self.n_clusters):
            results[c]["RI"] = np.max([results[c]["CS"][k] for k in range(len(self.class_to_idx.values()))])
        RIs_abs = [np.abs(results[c2]["RI"]) for c2 in range(self.n_clusters)]
        for c in range(self.n_clusters):
            results[c]["RI"] = results[c]["RI"]/np.max(RIs_abs)
        # compute class relative importance
        for c in range(self.n_clusters):
            results[c]["RIc"] = np.max([
                (results[c]["CS"][k]
                /
                np.max([np.abs(results[c2]["CS"][k]) for c2 in range(self.n_clusters)])
                ) for k in range(len(self.class_to_idx.values()))])
        self.results = results

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
        self.n_limit_images = configs["n_limit_images"]
        self.n_clusters = configs["n_clusters"]
        self.eclad_layers = configs["eclad_layers"]
        self.upsampling_mode = configs["upsampling_mode"]
        self.clustering_method = configs["clustering_method"]
        self.eclad_clustering_batches = configs["eclad_clustering_batches"]
        self.eclad_standardscale = configs["eclad_standardscale"]
        self.eclad_sample = configs["eclad_sample"]
        self.eclad_alpha = configs["eclad_alpha"]
        self.workers_activations = configs["workers_activations"]
        self.batch_size = configs["batch_size"]
        self.device = configs["device"]
        self.centroids = configs["centroids"]
        self.n_concept_examples = configs["n_concept_examples"]
        self.results = configs["results"]
        with open((Path(self.save_path)/"classifier.pkl").as_posix(),"rb") as f:
            self.classifier = pickle.load(f)
        with open((Path(self.save_path)/"scaler.pkl").as_posix(),"rb") as f:
            self.scaler = pickle.load(f)

    def save(self):
        config = {
            "class_to_idx": self.class_to_idx,
            "class_paths": self.class_paths,
            "shape": self.shape,
            "n_limit_images": self.n_limit_images,
            "n_clusters": self.n_clusters,
            "eclad_layers": self.eclad_layers,
            "upsampling_mode": self.upsampling_mode,
            "clustering_method": self.clustering_method,
            "eclad_clustering_batches": self.eclad_clustering_batches,
            "eclad_standardscale": self.eclad_standardscale,
            "eclad_sample": self.eclad_sample,
            "eclad_alpha": self.eclad_alpha,
            "workers_activations": self.workers_activations,
            "batch_size": self.batch_size,
            "device": self.device,
            "centroids": self.centroids,
            "n_concept_examples": self.n_concept_examples,
            "results": self.results
        }
        with open((Path(self.save_path)/"results.json").as_posix(), 'w') as f:
            json.dump(config, f, indent=4)
        with open((Path(self.save_path)/"classifier.pkl").as_posix(),"wb") as f:
            pickle.dump(self.classifier, f)
        with open((Path(self.save_path)/"scaler.pkl").as_posix(),"wb") as f:
            pickle.dump(self.scaler, f)

    def mask(self, image):
        # load image
        if isinstance(image, str): image = Image.open(image).convert('RGB')
        # eval image
        x = self.transforms(image).unsqueeze(0)
        # get lads
        LADs = self.get_lads(x).detach().cpu().numpy()
        # find concepts
        if self.eclad_standardscale:
            labels = self.classifier.predict(self.scaler.transform(LADs))
        else:
            labels = self.classifier.predict(LADs)
        # reshape
        mask = np.reshape(labels, self.shape)
        return mask

