"""
Model utils
"""


def count_model_params(model):
    """Counting number of learnable parameters"""
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params


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
