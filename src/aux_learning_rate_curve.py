"""
Computing and saving the learning rate curve. This helps visualize
the impact that learning rate warmups and schedulers will have on
the training procedure
"""

import os
import torch.nn as nn
from matplotlib import pyplot as plt

from lib.arguments import get_directory_argument
from lib.config import Config
from lib.schedulers import WarmupVSScehdule
import lib.setup_model as setup_model
import data


def main(exp_path):
    """ Main logic for computing the learning rate curve """
    exp_path = exp_path
    cfg = Config(exp_path)
    exp_params = cfg.load_exp_config_file()
    plots_path = os.path.join(exp_path, "plots")
    num_epochs = exp_params["training"]["num_epochs"]

    # loading data. We only need to know the number of batches per epoch
    batch_size = exp_params["training"]["batch_size"]
    dataset_name = exp_params["dataset"]["dataset_name"]
    train_set = data.load_data(dataset_name=dataset_name, split="train")
    train_loader = data.build_data_loader(dataset=train_set, batch_size=batch_size)
    num_iters = len(train_loader)

    # dummy model, optimizer and scheduler
    model = nn.Linear(2, 2)
    optimizer, scheduler, lr_warmup = setup_model.setup_optimizer(
            exp_params=exp_params,
            model=model
        )
    warmup_scheduler = WarmupVSScehdule(
            optimizer=optimizer,
            lr_warmup=lr_warmup,
            scheduler=scheduler
        )

    # computing intermediate learning rate values
    learning_rate_per_epoch, learning_rate_per_iter = compute_learning_rate_curve(
            exp_params=exp_params,
            optimizer=optimizer,
            warmup_scheduler=warmup_scheduler,
            num_epochs=num_epochs,
            num_iters=num_iters
        )

    # saving learning rate curves as plots
    scheduler = exp_params["training"]["scheduler"]
    lr_factor = exp_params["training"]["lr_factor"]
    patience = exp_params["training"]["patience"]
    scheduler_steps = exp_params["training"].get("scheduler_steps", 1e6)
    warmup = exp_params["training"]["lr_warmup"]
    warmup_steps = exp_params["training"]["warmup_steps"]
    warmup_epochs = exp_params["training"]["warmup_epochs"]
    savepath = os.path.join(plots_path, "learning_rate_curves.png")

    plt.style.use('seaborn')
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(12, 8)
    ax[0].plot(learning_rate_per_epoch)
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Learning Rate")
    ax[0].set_title("Learning Rate per Epoch")
    ax[1].plot(learning_rate_per_iter)
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("Learning Rate")
    ax[1].set_title("Learning Rate per Iter")
    fig.suptitle(
        f"Warmup: {warmup}  Steps {warmup_steps}  Epochs {warmup_epochs} \n"
        f"Scheduler: {scheduler}  LR-Factor: {lr_factor}  Patience: {patience}  Sch. Steps {scheduler_steps}"
    )
    plt.tight_layout()
    plt.savefig(savepath)
    return


def compute_learning_rate_curve(exp_params, optimizer, warmup_scheduler, num_epochs, num_iters):
    """ Iterating over epochs and iterations, saving intermediate learning rate values """
    learning_rate_per_iter = []
    learning_rate_per_epoch = []
    dummy_val = 1e6

    for epoch in range(num_epochs):
        for i in range(num_iters):
            cur_iter = epoch * num_iters + i
            warmup_scheduler(iter=cur_iter, epoch=epoch, exp_params=exp_params, end_epoch=False)
            learning_rate_per_iter.append(optimizer.param_groups[0]['lr'])
        warmup_scheduler(
                iter=-1,
                epoch=epoch,
                exp_params=exp_params,
                end_epoch=True,
                control_metric=dummy_val,
            )
        learning_rate_per_epoch.append(optimizer.param_groups[0]['lr'])

    return learning_rate_per_epoch, learning_rate_per_iter


if __name__ == '__main__':
    exp_path, args = get_directory_argument()
    main(exp_path)
