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
