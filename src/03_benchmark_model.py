"""
Benchmarking a model in terms of Throughput, FLOPS, Number of Parameters, ...
"""

import os
import json
import torch

from lib.arguments import get_directory_argument
from lib.config import Config
from lib.logger import Logger, print_, log_function, for_all_methods
import lib.setup_model as setup_model
import lib.utils as utils
import models.model_utils as model_utils
import data


@for_all_methods(log_function)
class Benchmarker:
    """
    Class for evaluating a model
    """

    def __init__(self, exp_path, checkpoint):
        """
        Initializing the trainer object
        """
        self.exp_path = exp_path
        self.cfg = Config(exp_path)
        self.exp_params = self.cfg.load_exp_config_file()
        self.checkpoint = checkpoint

        self.models_path = os.path.join(self.exp_path, "models")
        self.results_path = os.path.join(self.exp_path, "results")
        utils.create_directory(self.results_path)
        return

    def load_data(self):
        """ Loading dataset """
        self.dataset = data.load_data(exp_params=self.exp_params, split="train")
        return

    def setup_model(self):
        """ Initializing model and loading pretrained parameters """
        torch.backends.cudnn.fastest = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # loading model
        self.model = setup_model.setup_model(model_params=self.exp_params["model"])
        self.model = self.model.eval().to(self.device)

        checkpoint_path = os.path.join(self.models_path, self.checkpoint)
        self.model = setup_model.load_checkpoint(
                checkpoint_path=checkpoint_path,
                model=self.model,
                only_model=True
            )
        self.model = self.model.eval().to(self.device)
        print_("  --> Pretrained parameters loaded successfully")
        return

    @torch.no_grad()
    def benchmark(self):
        """ Benchmarking the model """

        print_("Benchmarking number of learnambe parameters...")
        num_params = model_utils.count_model_params(self.model, verbose=True)

        print_("Benchmarking FLOPS and #Activations...")
        img = self.dataset[0][0].to(self.device)
        total_flops, total_act = model_utils.compute_flops(
                model=self.model,
                dummy_input=img.unsqueeze(0),
                detailed=False,
                verbose=True
            )

        print_("Benchmarking Throughput...")
        throughput, avg_time_per_img = model_utils.compute_throughput(
                model=self.model,
                dataset=self.dataset,
                device=self.device,
                num_imgs=500,
                use_tqdm=True,
                verbose=True
            )

        print_("Saving benchmark results...")
        results = {
            "num_params": num_params,
            "total_flops": total_flops,
            "total_act": total_act,
            "throughput": throughput,
            "avg_time_per_img": avg_time_per_img
        }
        checkpoint_name = self.checkpoint.split(".")[0]
        savepath = os.path.join(self.results_path, f"benchmark_{checkpoint_name}.json")
        with open(savepath, "w") as f:
            json.dump(results, f)

        return


if __name__ == "__main__":
    utils.clear_cmd()
    exp_path, args = get_directory_argument()
    logger = Logger(exp_path=exp_path)
    logger.log_info("Starting benchmarking procedure", message_type="new_exp")
    logger.log_git_hash()

    print_("Initializing Evaluator...")
    benchmarker = Benchmarker(exp_path=exp_path, checkpoint=args.checkpoint)
    print_("Loading dataset...")
    benchmarker.load_data()
    print_("Setting up model and loading pretrained parameters")
    benchmarker.setup_model()
    benchmarker.benchmark()


#
