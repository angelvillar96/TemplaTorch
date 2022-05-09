# TemplaTorch

Simple, yet effective, template for quick deep learning prototyping and experimentation


## Contents

 * [1. Getting Started](#getting-started)
 * [2. Directory Structure](#directory-structure)
 * [3. Quick Guide](#quick-guide)
 * [4. Contact](#contact)


 ## Getting Started

 ### Prerequisites

 To get the repository running, you will need several python packages, e.g., ```PyTorch```, ```Numpy```, or ```Matplotlib```.

 You can install them all easily and avoiding dependency issues by installing the ```conda``` environment file included in the repository. To do so, run the following command from the terminal:

 ```shell
 $ conda env create -f environment.yml
 $ conda activate TemplaTorch
 ```


 ## Directory Structure

 The following tree diagram displays the detailed directory structure of the project. Directory names and paths can be modified in the [CONFIG File](https://github.com/angelvillar96/TemplaTorch/blob/master/src/CONFIG.py).

 ```
 TemplaTorch
 ├── experiments/
 |   ├── exp1/
 |   └── exp2/
 ├── src/
 |   └── ...
 ├── environment.yml
 └── README.md
 ```



 ## Quick Guide

 Follow this section for a quick guide on how to get started with this repository.

### Update Existing Codebase.

 By default, ```TemplaTorch``` includes a simple Convolutional Neural Network (CNN) to perform handwritten digit recognition, i.e., classification on the MNIST dataset.

 Most likely, your task is completely different. You can simply adapt the codebase by following the next steps:

   1. Add your model files to  `/src/models/`, and include your model in `/src/models/__init__.py`.

   2. Download your dataset to the data directory specified in `/src/CONFIG.py`.

   3. Write your Dataset class under `/src/data/`, and support loading it in `/src/data/load_data.py`

   4. Modifiy the configurations (`/src/CONFIG.py`) to support your model and data

   5. If necessary, change the training/eval loops in `/src/02_train.py` and `/src/03_evaluate.py`


### Creating an Experiment

```shell
$ python src/01_create_experiment.py [-h] -d EXP_DIRECTORY [--name NAME] [--config CONFIG]

optional arguments:
  -h, --help            show this help message and exit
  -d EXP_DIRECTORY, --exp_directory EXP_DIRECTORY Directory where the experiment folder will be created
  --name NAME           Name to give to the experiment
  --config CONFIG       Name of the predetermined 'config' to use
```

Creating an experiment automatically generates a directory in the specified EXP_DIRECTORY, containing a *JSON* file with the experiment parameters and sub-directories for the models, plot, and Tensorboard logs.


### Training and Evaluating a Model

Once the experiment is initialized and the experiment parameters are set to the desired values, a model can be trained following command:

```shell
$ CUDA_VISIBLE_DEVICES=0 python src/02_train.py [-h] -d EXP_DIRECTORY [--checkpoint CHECKPOINT] [--resume_training]

optional arguments:
  -h, --help            show this help message and exit
  -d EXP_DIRECTORY, --exp_directory EXP_DIRECTORY Path to the experiment directory
  --checkpoint CHECKPOINT Checkpoint with pretrained parameters to load
  --resume_training     For resuming training

```

Model checkpoints, which are saved regularly during training, can be evaluated using the following command:

```shell
$ CUDA_VISIBLE_DEVICES=0 python src/3_evaluate.py [-h] -d EXP_DIRECTORY [--checkpoint CHECKPOINT]

optional arguments:
  -h, --help            show this help message and exit
  -d EXP_DIRECTORY, --exp_directory EXP_DIRECTORY Path to the experiment directory
  --checkpoint CHECKPOINT Checkpoint with pretrained parameters to load
```


##### Example


```shell
$ python src/01_create_experiment.py -d TemplaTorch_Tests --name exp_mnist --config mnist_basic
$ CUDA_VISIBLE_DEVICES=0 python src/02_train.py -d experiments/TemplaTorch_Tests/exp_mnist/
$ CUDA_VISIBLE_DEVICES=0 python src/03_evaluate.py -d experiments/TemplaTorch_Tests/exp_mnist/ --checkpoint checkpoint_epoch_10.pth

```


### Hyper-Parameter Optimization

TemplaTorch includes functionalities for hyper-parameter optimization using the [Optuna](https://optuna.org/) framework.

```
CUDA_VISIBLE_DEVICES=0 python src/01_hyperparam_optim.py -[-h] -d EXP_DIRECTORY [--num_epochs NUM_EPOCHS] [--num_trials NUM_TRIALS]

optional arguments:
  -h, --help            show this help message and exit
  -d EXP_DIRECTORY, --exp_directory EXP_DIRECTORY
                        Path to the experiment directory
  --num_epochs NUM_EPOCHS
                        Number of epochs of each trial
  --num_trials NUM_TRIALS
                        Number of trials to execute

```

The file `CONFIG.py` defines the parameters to be optimized, including the selection ranges, data-type, among other arguments.
Any of the parameters in `experiment_parameters.json` can be optimized.

The procedure for hyper-parameter optimization in TemplaTorch is the following:

 1. Add the parameters that you want to optimize to the configuration (`CONFIG.py`) file.

 2. Run the hyper-param optimization scrip: ```python src/01_hyperparam_optim.py -d EXP_DIR```. This creates a directory already including an `experiment_parameters.json` and a `optuna_values.json` configuration files, which contain the default experiment parameters and the arguments for the parameters to optimize, respectively.

 3. Modify the values in `experiment_parameters.json` and `optuna_values.json` to match your needs. For instance, add your desired ranges for the *learning rate* and *batch size*.

 4. Run again the hyper-param optimization scrip: ```python src/01_hyperparam_optim.py -d EXP_DIR --num_epochs NUM_EPOCHS --num_trials NUM_TRIALS```, which will run *NUM_TRIALS* Optuna Trials, for *NUM_EPOCHS* each, saving intermediate results.

 5. The results of the hyper-parameter optimization study will be stored on a *pickle* file. Additionally, intermediate results for all trials, as well as an SQL database, are saved in the experiment directory.



 ##### Example

 ```shell
 $ CUDA_VISIBLE_DEVICES=0 python src/01_hyperparam_optim.py -d experiments/-d TemplaTorch_Tests/optuna_test/
 $ CUDA_VISIBLE_DEVICES=0 python src/01_hyperparam_optim.py -d experiments/-d TemplaTorch_Tests/optuna_test/
 ```

Results of an actual Optuna study can be found in [this directory](TODO).



## Contact

This repository is maintained by [Angel Villar-Corrales](http://angelvillarcorrales.com/templates/home.php),

In case of any questions or problems regarding the project or repository, do not hesitate to contact the authors at villar@ais.uni-bonn.de.
