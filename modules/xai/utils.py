import numpy as np
import torch
from PIL import Image
import torchvision
from pathlib import Path
from modules.activations.activations_pytorch import get_layers, get_gradients

def patch_from_mask(image, mask, alpha=-1):
    mask_expanded = np.expand_dims(mask, -1)
    if alpha==-1:
        patch = (mask_expanded * np.array(image) + (1 - mask_expanded) * np.mean(np.array(image))).astype(np.uint8)
    else:
        patch = (mask_expanded * np.array(image) + (1 - mask_expanded) * np.array(image) * alpha).astype(np.uint8)

    ones = np.where(mask == 1)
    h1, h2, w1, w2 = ones[0].min(), ones[0].max(), ones[1].min(), ones[1].max()
    h1, h2, w1, w2 = min(max(h1,0),mask.shape[0]), min(max(h2,0),mask.shape[0]), min(max(w1,0),mask.shape[1]), min(max(w2,0),mask.shape[1])
    if ((h2-h1)<=1) or ((w2-w1)<=1):
        superpixel = Image.fromarray(patch)
    else:
        superpixel = Image.fromarray((patch[h1:h2, w1:w2]))
    return superpixel, patch

class DummyDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, image_paths, transforms=None):
        'Initialization'
        self.image_paths = image_paths
        self.transforms = transforms

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        image_path = self.image_paths[index]
        # Load data
        image = Image.open(image_path).convert('RGB')
        # Load data and get label
        y = 0
        if self.transforms is None:
            X = torchvision.transforms.ToTensor()(image)
            #X = torch.tensor(image)
        else:
            X = self.transforms(image)
        return X, y

def encode_folder(model, images_path, results_path, layer, class_idx, transforms, workers=1, batch_size=4, device="cpu", namestring="", limit=-1):
    # if the folder was already encoded
    if (Path(results_path)/f"acts{namestring}.npy").exists() and (Path(results_path)/f"grads{namestring}_{class_idx}.npy").exists():
        filename=(Path(results_path)/f"acts{namestring}.npy").as_posix()
        acts_list = np.load(filename)
        filename=(Path(results_path)/f"grads{namestring}_{class_idx}.npy").as_posix()
        grads_list = np.load(filename)
        return acts_list, grads_list
    # glob images in folder
    image_paths = sorted([p for p in Path(images_path).glob("*") if p.suffix in [".jpg", ".jpeg", ".png", ".bmp", ".gif"]])
    if limit!=-1: image_paths = image_paths[:limit]
    ds = DummyDataset(image_paths, transforms)
    ds_loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=workers, shuffle=False)# fix batch gradient extraction
    layers = get_layers(model, names=[layer])
    acts_list, grads_list = [], []
    for xb,yb in ds_loader:
        outs, acts, grads = get_gradients(model, xb.to(device), layers, out_idx=class_idx)
        acts_list.extend(acts[layer].cpu().detach().numpy())
        grads_list.extend(grads[layer].cpu().detach().numpy())
    # save
    acts_list = np.stack(acts_list, axis=0)#np.concatenate(acts_list)
    filename=(Path(results_path)/f"acts{namestring}.npy").as_posix()
    np.save(filename, acts_list)
    grads_list = np.stack(grads_list, axis=0)
    filename=(Path(results_path)/f"grads{namestring}_{class_idx}.npy").as_posix()
    np.save(filename, grads_list)
    return acts_list, grads_list