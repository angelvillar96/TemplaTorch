"""
Model utils
"""

from time import time
from tqdm import tqdm
import numpy as np
import torch
from fvcore.nn import FlopCountAnalysis, flop_count_str, ActivationCountAnalysis

from lib.logger import print_, log_info


def count_model_params(model, verbose=False):
    """Counting number of learnable parameters"""
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print_(f"  --> Number of learnable parameters: {num_params}")
    return num_params


def compute_flops(model, dummy_input, verbose=True, detailed=False):
    """ Computing the number of activations and flops in a forward pass """
    func = print_ if verbose else log_info

    # benchmarking
    fca = FlopCountAnalysis(model, dummy_input)
    act = ActivationCountAnalysis(model, dummy_input)
    if detailed:
        fcs = flop_count_str(fca)
        func(fcs)
    total_flops = fca.total()
    total_act = act.total()

    # logging
    func("  --> Number of FLOPS in a forward pass:")
    func(f"   --> FLOPS = {total_flops}")
    func(f"    --> FLOPS = {round(total_flops / 1e9, 3)}G")
    func("  --> Number of activations in a forward pass:")
    func(f"    --> Activations = {total_act}")
    func(f"    --> Activations = {round(total_act / 1e6, 3)}M")
    return total_flops, total_act


def compute_throughput(model, dataset, device, num_imgs=500, use_tqdm=True, verbose=True):
    """ Computing the throughput of a model in imgs/s """
    times = []
    N = min(num_imgs, len(dataset))
    iterator = tqdm(range(N)) if use_tqdm else range(N)
    model = model.to(device)

    # benchmarking by averaging over N images
    for i in iterator:
        img = dataset[i][0].unsqueeze(0).to(device)
        torch.cuda.synchronize()
        start = time()
        _ = model(img)
        torch.cuda.synchronize()
        times.append(time() - start)
    avg_time_per_img = np.mean(times)
    throughput = 1 / avg_time_per_img

    # logging
    func = print_ if verbose else log_info
    func(f"  --> Average time per image: {round(avg_time_per_img, 3)}s")
    func(f"  --> Throughput: {round(throughput)} imgs/s")
    return throughput, avg_time_per_img


def freeze_params(model):
    """Freezing model params to avoid updates in backward pass"""
    for param in model.parameters():
        param.requires_grad = False
    return model


def unfreeze_params(model):
    """Unfreezing model params to allow for updates during backward pass"""
    for param in model.parameters():
        param.requires_grad = True
    return model


class GradientInspector:
    """
    Module that computes some statistics from the gradients of one parameter,
    and logs the stats into the Tensorboard

    Args:
    -----
    writer: TensorboardWriter
        TensorboardWriter object used to log into the Tensorboard
    layers: list of nn.Module
        Layers whose gradients are processed and logged into the Tensorboard
    names: list of strings
        Name given to each of the layers to track
    stats: list
        List with the stats to track. Possible stats are: ['Min', 'Max', 'Mean', 'Var', 'Norm']
    """

    STATS = ["Min", "Max", "Mean", "MeanAbs", "Var", "Norm"]
    FUNCS = {
        "Min": torch.min,
        "Max": torch.max,
        "Mean": torch.mean,
        "MeanAbs": lambda x: x.abs().mean(),
        "Var": torch.var,
        "Norm": torch.norm,
    }

    def __init__(self, writer, layers, names, stats=None):
        """ Module initializer """
        stats = stats if stats is not None else GradientInspector.STATS
        for stat in stats:
            assert stat in GradientInspector.STATS, f"{stat = } not included in {self.STATS = }"
        assert isinstance(layers, list), f"Layers is not list, but {type(layers)}..."
        assert len(layers) == len(names), f"{len(layers) = } and {len(names) = } must be the same..."
        for layer in layers:
            assert isinstance(layer, torch.nn.Module), f"Layer is not nn.Module, but {type(layer)}..."
            assert hasattr(layer, "weight"), "Layer does not have attribute 'weight'"

        self.writer = writer
        self.layers = layers
        self.names = names
        self.stats = stats

        print("Initializing Gradient-Inspector:")
        print(f"  --> Tracking stats {stats} of gradients in the following layers")
        for name, layer in zip(names, layers):
            print(f"    --> {name}: {layer}")
        return

    def __call__(self, step):
        """ Computing gradient stats and logging into Tensorboard """
        for layer, name in zip(self.layers, self.names):
            grad = layer.weight.grad
            for stat in self.stats:
                func = self.FUNCS[stat]
                self.writer.add_scalar(f"Grad Stats {name}/{stat} Grad", func(grad).item(), step)
        return
