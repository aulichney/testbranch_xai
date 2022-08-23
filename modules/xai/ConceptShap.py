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
from torch.nn import functional as F
from torch import nn
import torchmetrics
from sklearn.cluster import Birch, MiniBatchKMeans, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from torchvision import transforms

from modules.utils.loggers import create_logger
from modules.activations.activations_pytorch import get_layers, get_gradients, get_activations
from modules.xai.utils import DummyDataset

class conceptShapModel(nn.Module):
    def __init__(self, n_concepts, n_features, B_threshold):
        super(conceptShapModel, self).__init__()
        self.n_concepts = n_concepts
        self.n_features = n_features
        self.B_threshold = B_threshold

        self.g = nn.Sequential(
                nn.Linear(n_concepts, 500),
                nn.ReLU(),
                nn.Linear(500, n_features)
            )
        self.C = nn.Linear(n_features, n_concepts, bias=False)

    def forward(self, x):
        g_vc_flat = self.g(x)
        return g_vc_flat

    def vc(self, x):
        vc_flat = self.C(x)
        vc_flat = nn.Threshold(self.B_threshold, 0, inplace=False)(vc_flat)
        vc_flat = F.normalize(vc_flat, p=2.0, dim=1)
        return vc_flat

class ConceptShap:
    def __init__(self,
        save_path,
        model,
        transforms,
        train_loader,
        val_loader,
        class_to_idx=None,
        class_paths=None,
        shape=(224,224),
        n_limit_images = 100,

        n_concepts = 10,
        n_epochs=10,
        cshap_layer = None,
        K = None,
        B_threshold = 0.2,
        cshap_optim_lr=0.01,
        lambda_1=0.1,
        lambda_2=0.1,
        MC_shapely_samples=20,
        completeness_tune_epochs=5,
        cshap_alpha=0.5,

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
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.class_to_idx = class_to_idx
        self.class_paths = class_paths
        self.shape = shape
        self.n_limit_images = n_limit_images

        self.n_concepts = n_concepts
        self.n_epochs = n_epochs
        self.cshap_layer = cshap_layer
        self.K = K
        self.B_threshold = B_threshold
        self.cshap_optim_lr = cshap_optim_lr
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.MC_shapely_samples = MC_shapely_samples
        self.completeness_tune_epochs = completeness_tune_epochs
        self.cshap_alpha = cshap_alpha

        self.workers_activations = workers_activations
        self.batch_size = batch_size
        self.device = device
        self.n_concept_examples = n_concept_examples
        self.n_features = None

        if logger is None: logger = create_logger("ConceptShap")
        self.logger= logger
        if rng is None: self.rng = np.random.default_rng(0)
        elif isinstance(rng, int): self.rng = np.random.default_rng(rng)
        else: self.rng = rng
        self.cshap_model = None
        self.cshap_concepts = None
        self.results = []
        self.model.to(self.device)

    def __call__(self):
        self.logger.debug("Starting ConceptShap.")
        # create result folder
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        # learn concepts
        self.logger.debug("ConceptShap: Learning concepts.")
        self.init_cshap_model()
        self.learn_concepts(epochs=self.n_epochs,log=True)
        self.logger.debug("ConceptShap: Learning concepts finished.")
        # extract examples
        self.logger.debug("ConceptShap: Generating examples started.")
        for class_name, class_idx in self.class_to_idx.items():
            # create folder for examples
            class_path = (Path(self.save_path)/class_name).as_posix()
            Path(class_path).mkdir(parents=True, exist_ok=True)
            # generate examples for images of this class
            class_images = [p.as_posix() for p in Path(self.class_paths[class_name]).glob("*") if p.suffix in [".jpg", ".jpeg", ".png", ".bmp", ".gif"]]
            self.generate_examples(class_images[:self.n_limit_images], save_path = class_path, n = self.n_concept_examples)
        self.logger.debug("ConceptShap: Generating examples finished.")
        # test concepts
        self.logger.debug("ConceptShap: Testing concepts started.")
        self.evaluate_importance()
        self.logger.debug("ConceptShap: Testing concepts finished.")
        # save results
        self.save()
        self.clean()
        self.logger.debug("Finishing ConceptShap.")

    def init_cshap_model(self):
        # get size of features
        if self.n_features is None:
            path_classes_images = [[p.as_posix() for p in Path(class_path).glob("*") if p.suffix in [".jpg", ".jpeg", ".png", ".bmp", ".gif"]] for class_path in self.class_paths.values()]
            image = Image.open(path_classes_images[0][0])
            x = self.transforms(image)
            x = x.unsqueeze(0).to(self.device)
            self.model = self.model.to(self.device)
            layers = get_layers(self.model,names=self.cshap_layer)
            out, act_store = get_activations(self.model, x, layers)
            self.n_features = act_store[self.cshap_layer].shape[1]
            self.logger.debug(f"shape of activation map: {act_store[self.cshap_layer].shape}")
        # create concept vector and model
        self.cshap_model = conceptShapModel(self.n_concepts, self.n_features, self.B_threshold)
        self.cshap_model = self.cshap_model.to(self.device)
        self.cshap_concepts = self.cshap_model.C.weight.detach().cpu().numpy().tolist()


    def evalConceptShapModel(self, x):
        self.model.to(self.device)
        self.cshap_model.to(self.device)
        layers = get_layers(self.model, names=self.cshap_layer)
        act_store={}
        vc_store={}
        g_vc_store={}
        # hook generate for cshap_model execution
        def hook_generator(name, cshap_model):
            def conceptShap_hook(model, input, output):
                # flatten activation map
                a = output
                a_flat = a.permute(0,2,3,1).flatten(0,2)
                # compute vc(a)
                vc_flat = cshap_model.vc(a_flat)
                # compute g(vc(a))
                g_vc_flat = cshap_model(vc_flat)
                # reshape
                a_shape = a.shape
                vc = vc_flat.unflatten(0,[a_shape[0],a_shape[2],a_shape[3]]).permute(0,3,1,2)
                g_vc = g_vc_flat.unflatten(0,[a_shape[0],a_shape[2],a_shape[3]]).permute(0,3,1,2)
                # save in global
                act_store[name]=a
                vc_store[name]=vc
                g_vc_store[name]=g_vc
                # return (this creates the new output of the layer)
                return g_vc
            return conceptShap_hook
        # create and attach the hooks
        handlers = {}
        for name, layer in layers.items():
            handlers[name] = layer.register_forward_hook(hook_generator(name, self.cshap_model))
        # eval
        out = self.model(x.to(self.device))
        # delete hooks
        for handler in handlers.values():
            handler.remove()
        return out, act_store, vc_store, g_vc_store

    def learn_concepts(self, lr=None, epochs=50, log=False):
        if lr is None: lr = self.cshap_optim_lr
        self.cshap_model.train()
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.K = self.train_loader.batch_size/(2*len(self.class_to_idx.values()))
        # init optimization
        criterion = torch.nn.NLLLoss()
        optimizer = torch.optim.SGD(self.cshap_model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3, threshold=0.00001, mode='min')
        lambda_1_adj = None
        lambda_2_adj = None
        # iterate train
        for epoch in range(epochs):
            losses = []
            train_acc = torchmetrics.Accuracy()
            train_cm = torchmetrics.ConfusionMatrix(num_classes=len(self.class_to_idx.values()))
            train_f1 = torchmetrics.F1Score(num_classes=len(self.class_to_idx.values()))
            for batch_idx, (data, target) in enumerate(self.train_loader):
                x, y = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                # model loss
                logits, act_store, vc_store, g_vc_store = self.evalConceptShapModel(x)
                y_hat_prob = F.softmax(logits, dim=-1)
                loss = criterion(logits, y)
                # R(c)
                activations = act_store[self.cshap_layer]
                a_flat = act_store[self.cshap_layer].permute(0,2,3,1).flatten(0,2)
                c_norm = F.normalize(self.cshap_model.C.weight, p=2.0, dim=1)
                a_norm = F.normalize(a_flat, p=2.0, dim=1)
                loss_proj = torch.sum(c_norm @ a_flat.T)/(self.K*self.n_concepts) # modified to avoid non unitary concept vectors
                loss_novelty = torch.tril((c_norm @ c_norm.T), diagonal=-1).sum()/(self.n_concepts*(self.n_concepts-1))
                loss -= self.lambda_1 * loss_proj - self.lambda_2 * loss_novelty
                # backward
                loss.backward()
                optimizer.step()
                losses.append(loss.detach().cpu().numpy())

                train_acc(y_hat_prob.cpu(), y.cpu())
                train_cm(y_hat_prob.cpu(), y.cpu())
                train_f1(y_hat_prob.cpu(), y.cpu())
            scheduler.step(loss.mean())

            if log: self.logger.debug(f"Epoch {epoch}: loss={np.mean(losses):.3E} acc={train_acc.compute():.3E} f1={train_f1.compute():.3E} lr:{optimizer.param_groups[0]['lr']:.2E}")
            if optimizer.param_groups[0]['lr']<lr*10e-8:
                break
        # remove overlapping concepts (as per the paper)
        c_norm = F.normalize(self.cshap_model.C.weight, p=2.0, dim=1)
        similarity = torch.tril((c_norm @ c_norm.T), diagonal=-1).detach().cpu().numpy()
        removed = []
        for c1 in range(self.n_concepts):
            for c2 in range(c1+1, self.n_concepts):
                if (similarity[c2, c1]>0.95) and (c1 not in removed):
                    S = [1 if i!=c2 else 0 for i in range(self.n_concepts)]
                    self.cshap_model.C.weight.data = \
                            (self.cshap_model.C.weight.clone().detach().to(self.device) *\
                            torch.tensor(S).unsqueeze(-1).to(self.device)).data
                    self.logger.debug(f"Removing similar concepts: c_{c1:02}-c_{c2:02}={similarity[c2,c1]}")
                    removed.append(c2)
        self.cshap_concepts = self.cshap_model.C.weight.detach().cpu().tolist()

    def generate_examples(self, image_paths, save_path, n=10):
        Path(save_path).mkdir(parents=True, exist_ok=True)
        counters = [n for i in range(self.n_concepts)]
        for img_path in image_paths:
            # open image
            image = Image.open(img_path).convert('RGB')
            # mask
            mask = self.mask(image)
            image = image.resize(self.shape)
            mask = [m[..., np.newaxis] for m in mask]
            # save examples
            for concept_idx in range(self.n_concepts):
                (Path(save_path)/f"c_{concept_idx:02}").mkdir(parents=True, exist_ok=True)
                if (counters[concept_idx]>0) and (np.sum(mask[concept_idx])>0):
                    # save masked image
                    masked_image = (np.array(image)*(mask[concept_idx]==True) + np.array(image)*(mask[concept_idx]==False)*self.cshap_alpha).astype(np.uint8)
                    masked_image = Image.fromarray(masked_image)
                    save_name = (Path(save_path)/f"c_{concept_idx:02}"/Path(img_path).name).as_posix()
                    masked_image.save(save_name)
                    # update counters
                    counters[concept_idx] = counters[concept_idx]-1
            # check if all examples were found
            if sum(counters)==0:
                break

    def evaluate_completeness(self, acc_model=None, acc_rand=None, tune=0):
        # This method receives balanced datasets
        train_acc = torchmetrics.Accuracy()
        if acc_model is None: train_acc_model = torchmetrics.Accuracy()
        if acc_rand is None: acc_rand = 1.0/len(self.class_to_idx.values())
        # fine tune g here
        self.cshap_model.C.weight.requires_grad = False
        self.learn_concepts(epochs=tune)
        # evaluate
        self.cshap_model.eval()
        self.model.eval()
        for batch_idx, (data, target) in enumerate(self.val_loader):
            if batch_idx*self.batch_size>self.n_limit_images: break
            x, y = data.to(self.device), target.to(self.device)
            # concept model
            concept_logits, _, _, _ = self.evalConceptShapModel(x)
            y_hat_prob = F.softmax(concept_logits, dim=-1)
            train_acc(y_hat_prob.cpu(), y.cpu())
            if acc_model is None:
                logits = self.model(x)
                y_hat_prob = F.softmax(logits, dim=-1)
                train_acc_model(y_hat_prob.cpu(), y.cpu())

        acc_concepts = train_acc.compute().tolist()
        if acc_model is None: acc_model = train_acc_model.compute().tolist()
        completeness = (acc_concepts - acc_rand) / (acc_model - acc_rand)
        return completeness, acc_model, acc_concepts, acc_rand

    def evaluate_importance(self):
        # get layers
        layers = get_layers(self.model, names=self.cshap_layer)
        self.cshap_concepts = self.cshap_model.C.weight.detach().cpu().tolist()
        torch.save(self.cshap_model.state_dict(), (Path(self.save_path)/"CShap_model.ckpt").as_posix())
        checkpoint = torch.load((Path(self.save_path)/"CShap_model.ckpt").as_posix())
        # evaluate completeness
        self.completeness, acc_model, acc_concepts, acc_rand = self.evaluate_completeness(acc_model=None)
        # evaluate importance
        S_completeness_list = {}
        S_completeness_list[str([1]*self.n_concepts)] = self.completeness
        Sh = {i:{} for i in range(self.n_concepts)}
        for n_sample in range(self.MC_shapely_samples):
            idx = list(range(self.n_concepts))
            self.rng.shuffle(idx)
            S = [0]*self.n_concepts
            for sub_idx in range(self.n_concepts):
                S_i = [1 if i in idx[:sub_idx+1] else 0 for i in range(self.n_concepts)]
                # evaluate S
                if str(S) in S_completeness_list:
                    S_completeness = S_completeness_list[str(S)]
                else:
                    self.cshap_model.load_state_dict(checkpoint)
                    self.cshap_model.C.weight.data = \
                        (self.cshap_model.C.weight.clone().detach().to(self.device) *\
                        torch.tensor(S).unsqueeze(-1).to(self.device)).data
                    S_completeness_list[str(S)], _, _, _ = self.evaluate_completeness(acc_model=acc_model, tune=self.completeness_tune_epochs)
                # evaluate S_i
                if str(S_i) in S_completeness_list:
                    S_completeness = S_completeness_list[str(S_i)]
                else:
                    self.cshap_model.load_state_dict(checkpoint)
                    if torch.norm(self.cshap_model.C.weight, p=2, dim=1).detach().cpu().numpy()[idx[sub_idx]]==0:
                        # if the norm of the concept weights is 0, it means we removed it beforehand (duplication), and thus, it's score is the same as S
                        S_completeness_list[str(S_i)] = S_completeness_list[str(S)]
                    else:
                        self.cshap_model.C.weight.data = \
                            (self.cshap_model.C.weight.clone().detach().to(self.device) *\
                            torch.tensor(S_i).unsqueeze(-1).to(self.device)).data
                        S_completeness_list[str(S_i)], _, _, _ = self.evaluate_completeness(acc_model=acc_model, tune=self.completeness_tune_epochs)
                # save iteration of MC Shap
                Sh[idx[sub_idx]][str(S+S_i)] = S_completeness_list[str(S_i)]-S_completeness_list[str(S)]
                S = S_i.copy()
            Sh_mean = {i:f"{np.mean(list(Sh[i].values())):.2f}~{np.std(list(Sh[i].values())):.2f}" for i in range(self.n_concepts)}
            self.logger.debug(f"MC shap [{n_sample}]:{Sh_mean}")
            self.results =[{
                "c_vec": self.cshap_concepts[k],
                "shap_mean": np.mean(list(v.values())),
                "shap_std": np.std(list(v.values())),
                "shap_MC":v,
                "concept_paths": {class_name:(Path(self.save_path)/class_name/f"c_{k:02}").as_posix()for class_name in self.class_to_idx.keys()},
                "idx": k,
                "name": f"c_{k:02}"
            } for k,v in Sh.items()]
        # correct Concepts
        self.cshap_model.load_state_dict(checkpoint)
        self.cshap_model.to(self.device)

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
        self.n_concepts = configs["n_concepts"]
        self.n_epochs = configs["n_epochs"]
        self.cshap_layer = configs["cshap_layer"]
        self.K = configs["K"]
        self.B_threshold = configs["B_threshold"]
        self.cshap_optim_lr = configs["cshap_optim_lr"]
        self.lambda_1 = configs["lambda_1"]
        self.lambda_2 = configs["lambda_2"]
        self.MC_shapely_samples = configs["MC_shapely_samples"]
        self.completeness_tune_epochs = configs["completeness_tune_epochs"]
        self.cshap_alpha = configs["cshap_alpha"]
        self.workers_activations = configs["workers_activations"]
        self.batch_size = configs["batch_size"]
        self.device = configs["device"]
        self.n_concept_examples = configs["n_concept_examples"]
        self.results = configs["results"]
        self.cshap_concepts = configs["cshap_concepts"]
        self.cshap_alpha = configs["cshap_alpha"]
        self.n_features = configs["n_features"]
        # init model
        self.init_cshap_model()
        # load model weights
        if (Path(self.save_path)/"CShap_model.ckpt").exists():
            checkpoint = torch.load((Path(self.save_path)/"CShap_model.ckpt").as_posix())
            self.cshap_model.load_state_dict(checkpoint)
            self.cshap_model = self.cshap_model.to(self.device)
            self.cshap_concepts = self.cshap_model.C.weight.detach().cpu().tolist()

    def save(self):
        config = {
            "class_to_idx": self.class_to_idx,
            "class_paths": self.class_paths,
            "shape": self.shape,
            "n_limit_images": self.n_limit_images,
            "n_concepts" : self.n_concepts,
            "n_epochs": self.n_epochs,
            "cshap_layer" : self.cshap_layer,
            "K" : self.K,
            "B_threshold" : self.B_threshold,
            "cshap_optim_lr" : self.cshap_optim_lr,
            "lambda_1" : self.lambda_1,
            "lambda_2" : self.lambda_2,
            "MC_shapely_samples": self.MC_shapely_samples,
            "completeness_tune_epochs": self.completeness_tune_epochs,
            "cshap_alpha": self.cshap_alpha,
            "workers_activations": self.workers_activations,
            "batch_size": self.batch_size,
            "device": self.device,
            "n_concept_examples": self.n_concept_examples,
            "results": self.results,
            "cshap_concepts": self.cshap_concepts,
            "cshap_alpha": self.cshap_alpha,
            "n_features": self.n_features
        }

        with open((Path(self.save_path)/"results.json").as_posix(), 'w') as f:
            json.dump(config, f, indent=4)
        if self.cshap_model is not None:
            torch.save(self.cshap_model.state_dict(), (Path(self.save_path)/"CShap_model.ckpt").as_posix())

    def mask(self, image, threshold=-1):
        # load image
        if isinstance(image, str): image = Image.open(image).convert('RGB')
        # eval image
        x = self.transforms(image).unsqueeze(0).to(self.device)
        logits, act_store, vc_store, g_vc_store = self.evalConceptShapModel(x)
        y_hat_prob = F.softmax(logits, dim=-1)
        # masks
        masks = vc_store[self.cshap_layer].clone().cpu().detach()
        masks = F.interpolate(masks, size=self.shape, mode="nearest")
        if threshold ==-1:
            masks = [(masks[0,c,:,:].numpy()>masks[0,c,:,:].numpy().mean()) for c in range(self.n_concepts)]
        else:
            masks = [(masks[0,c,:,:].numpy()>threshold) for c in range(self.n_concepts)]
        return masks
