import torch
from PIL import Image
from torch.autograd import grad
import torchvision.transforms as transforms
from torch import nn

def get_layers(model, names=[]):
    """
    Get a dict containing all layers in a pytorch model.

    Args:
        model (torch module): torch module which will be iterated searching for layers.
        names (list, optional): list of names of the layers to extract. if empty, all layers will be extracted. Defaults to [].

    Returns:
        dict: dictionary containing the name and reference to each layer in a model.
    """
    if not isinstance(names,list): names=[names]
    layers = {}
    for name, layer in model.named_modules():
        # if the module does not have children, then, it's a "layer"
        if len(list((layer.children())))==0: 
            if len(names)==0 or name in names:
                layers[name]=layer
    return layers

def get_activations(model, x, layers):   
    """
    Evaluate an input x on a model and extract the results from a dict of layers.

    Args:
        model (torch module): pytorch model to evaluate.
        x (torch tensor): input torch tensor.
        layers (dict): dictionary with the references to the layers in the model.

    Returns:
        out (torch tensor), act_store (dict): output of evaluating the model, and dict containing all activation maps for the selected layers.
    """
    act_store={}
    # hook generate for activations
    def hook_generator(name):
        def get_activations_hook(model, input, output):
            # we save activations on a global variable named act_store. 
            # This is probably a horrible design desicion, but it works.
            act_store[name] = output
        return get_activations_hook
    # create and attach the hooks
    handlers = {}
    for name, layer in layers.items():
        handlers[name] = layer.register_forward_hook(hook_generator(name))
    # eval
    model.zero_grad()
    out = model(x)
    # delete hooks
    for handler in handlers.values():
        handler.remove()
    return out, act_store

def get_gradients(model, x, layers, out_idx):
    """
    Evaluate an input x on a model and extract the results and gradient from a dict of layers.

    Args:
        model (torch module): pytorch model to evaluate.
        x (torch tensor): input torch tensor.
        layers (dict): dictionary with the references to the layers in the model.
        out_idx (int): index of output for computing the gradient.

    Returns:
        out (torch tensor), act_store (dict), grads_store (dict): output of evaluating the model, dict containing all activation maps for the selected layers, and dict containing the gradient of each one of the activation maps.
    """
    act_store={}
    grads_store={}
    # hook generate for activations
    def hook_generator(name):
        def get_activations_hook(model, input, output):
            # we save activations on a global variable named act_store. 
            # This is probably a horrible design desicion, but it works.
            act_store[name] = output
        return get_activations_hook
    # create and attach the hooks
    handlers = {}
    for name, layer in layers.items():
        handlers[name] = layer.register_forward_hook(hook_generator(name))
    # eval
    model.zero_grad()
    outs = model(x)
    for name, act in act_store.items():
        # compute the gradient for each activation map
        grads_store[name]=grad(outs[:,out_idx], act_store[name], grad_outputs=torch.ones_like(outs[:,out_idx]), retain_graph=True)[0]
    # delete hooks
    for handler in handlers.values():
        handler.remove()
    return outs, act_store, grads_store