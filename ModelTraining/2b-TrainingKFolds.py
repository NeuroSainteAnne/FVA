######### Script for Training with K-Fold Cross-Validation #########
# J. Benzakoun 2024
#
# Usage:
#   python 2b-TrainingKFolds.py [-n NAME] [-ram PRELOAD_RAM] [-w WANDB_PROJECT]
#
# Arguments:
#   -n, --name            : Name for the experiment/run (default: "FVV")
#   -ram, --preload_RAM    : Preload data into RAM, set to enable (default: False)
#   -w, --wandb_project    : Weights & Biases project name for logging (default: "FVV")
#
######################################################################

import argparse
import os
import numpy as np
import subprocess

# Default settings for project name and model name
defaultname = "FVV"
default_wandb_project = "FVV"

# Load dimensions of dataset from a saved file
with open(os.path.join("preprocessed_data","dimensions.txt"), "r") as file:
    z_dim = int(file.read())

# Load metadata from memory-mapped file
metadat = np.memmap(os.path.join("preprocessed_data","meta.memmap"), 
                         dtype=np.int32, mode="r", shape=(z_dim,5))

if __name__ == "__main__":
    # Argument parser for running the script from the command line
    parser = argparse.ArgumentParser(description="DeepLearner",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-n", "--name", help="name", default=defaultname)
    parser.add_argument("-ram", "--preload_RAM", help="preload_RAM", default=None)
    parser.add_argument("-w", "--wandb_project", help="wandb project name", default=default_wandb_project)
    name = vars(parser.parse_args())["name"]
    preload_RAM = vars(parser.parse_args())["preload_RAM"]
    wandb_project = vars(parser.parse_args())["wandb_project"]

    # Determine the number of k-folds based on metadata
    kfolds = np.max(metadat[:,3])+1

    # Iterate through each k-fold and run the training script for each fold
    for k in range(kfolds):
        # Create a call list to run the training script "2a-Training.py" with the specified arguments
        call = ["python", "2a-Training.py", "--name", name]
        if preload_RAM is not None:
            call += ["--preload_RAM", preload_RAM]
        call += ["--wandb_project", wandb_project, "--kfold", str(k)]
        # Execute the training script as a separate process for the current k-fold
        subprocess.run(call)