import argparse
import os
import numpy as np
import subprocess

# Default settings for project name and model name
defaultname = "FVV"
default_wandb_project = "FVV"

list_models = [
            "saved_models/janequin-k0-epoch27.pth",
            "saved_models/janequin-k1-epoch27.pth",
            "saved_models/janequin-k2-epoch29.pth",
            "saved_models/janequin-k3-epoch50.pth",
            "saved_models/janequin-k4-titan-epoch47.pth",
            "saved_models/janequin-k5-epoch33.pth",
            "saved_models/janequin-k6-epoch40.pth",
            "saved_models/janequin-k7-epoch35.pth",
            "saved_models/janequin-k8-epoch44.pth",
            "saved_models/janequin-k9-epoch43.pth",
        ]


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
    kfolds = 10 # np.max(metadat[:,3])+1

    # Iterate through each k-fold and run the training script for each fold
    for k in range(1,kfolds):
        # Create a call list to run the training script "2a-Training.py" with the specified arguments
        call = ["python", "2a-Training.py", "--name", name]
        ##call += ["--debug", "True"]
        if preload_RAM is not None:
            call += ["--preload_RAM", preload_RAM]
        call += ["--wandb_project", wandb_project, "--kfold", str(k), "--preload_model", list_models[k], 
               "--compatibility_mode", "True", "--learning_rate", str(0.0001), "--num_epochs", str(100) ]
        call += ["--qat_finetune", "True"]
        # Execute the training script as a separate process for the current k-fold
        subprocess.run(call)