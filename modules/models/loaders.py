from inspect import isclass
import numpy as np
import PIL
import torch
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
import torchvision
from sklearn.model_selection import train_test_split

# build transforms
def build_transforms(
    config_list=[
            {"type":"Resize", "size":(224,224)},
            {"type":"ColorJitter", "hue":.05, "saturation":.05},
            {"type":"RandomHorizontalFlip",},
            {"type":"RandomRotation","degrees":20},
            {"type":"ToTensor",},
            {"type":"Normalize","mean":[0.485, 0.456, 0.406], "std":[0.229, 0.224, 0.225]},
        ]
        ):
    transforms=[]
    for t_config in config_list:
        if t_config["type"]=="RandomRotation":
            t_config["resample"]=PIL.Image.BILINEAR
        transforms.append(getattr(torchvision.transforms, t_config["type"])(**{k:v for k,v in t_config.items() if k!="type"}))
    transforms = torchvision.transforms.Compose(transforms)
    return transforms

# build loader
def build_classification_loaders(dataset_base_path, dataloader_config, transform_train_config={}, transform_val_config={}, train_val_split=0.85, size_multiplier=2.0, shuffle=True, rng=None):
    if isinstance(rng, int): np.random.default_rng(rng)
    if rng is None: rng = np.random.default_rng(0)
    # build transforms
    transforms_train = build_transforms(**transform_train_config)
    transforms_val = build_transforms(**transform_val_config)
    
    # build dataset
    dataset_train = torchvision.datasets.ImageFolder(root=dataset_base_path, transform=transforms_train)
    dataset_val = torchvision.datasets.ImageFolder(root=dataset_base_path, transform=transforms_val)
    
    # create splits samplers
    w_train, w_test = make_weights_for_balanced_classes(dataset_train.targets, len(dataset_train.classes), train_val_split=train_val_split, shuffle=shuffle)

    train_sampler = WeightedRandomSampler(w_train, int(len(w_train)*size_multiplier))
    val_sampler = WeightedRandomSampler(w_test, int(len(w_test)*size_multiplier))
    
    # create loaders
    train_loader = torch.utils.data.DataLoader(dataset_train, sampler=train_sampler, **dataloader_config)
    val_loader = torch.utils.data.DataLoader(dataset_val, sampler=val_sampler, **dataloader_config)
    return train_loader, val_loader

def make_weights_for_balanced_classes(targets, nclasses, train_val_split=0.8, shuffle=True):
    idx_classes = [[i for i in range(len(targets)) if targets[i]==c] for c in range(nclasses)]
    # split
    w_train = [0]*len(targets)
    w_test = [0]*len(targets)
    for samples in idx_classes:
        s_train, s_test = train_test_split(samples, shuffle=shuffle, train_size=train_val_split, random_state=0)
        for s in s_train:
            w_train[s] = 1.0/len(s_train)
        for s in s_test:
            w_test[s] = 1.0/len(s_test)
    w_train = np.array(w_train)/np.sum(w_train)
    w_test = np.array(w_test)/np.sum(w_test)
    return w_train, w_test