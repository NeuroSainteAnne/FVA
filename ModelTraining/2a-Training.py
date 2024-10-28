######### Script for Training Deep Learning Model (1 fold) #########
# J. Benzakoun 2024
#
# Usage:
#   python 2a-Training.py [-k KFOLD] [-n NAME] [-d DEBUG] [-ram PRELOAD_RAM] [-w WANDB_PROJECT]
#
# Arguments:
#   -k, --kfold           : Fold index for k-fold cross-validation (default: 0)
#   -n, --name            : Name for the experiment/run (default: "FVV")
#   -d, --debug           : Debug mode, set to enable debugging (default: False)
#   -ram, --preload_RAM   : Preload data into RAM, set to enable (default: False)
#   -w, --wandb_project   : Weights & Biases project name for logging (default: "FVV")
#
######################################################################

import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torcheval.metrics import BinaryAccuracy
import psutil
import argparse
import sys
sys.path.insert(0,os.path.join(os.getcwd(),"../dependencies/smp3d"))
import segmentation_models_pytorch_3d as smp3d
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import warnings
import json
import wandb

from modules.generator import FVVDataset
from modules.model import create_FVV_model
from modules.figure import create_figure
from modules.loops import validation_loop, training_loop, bn_loop
from modules.classes import TrainingObject

### Default settings for project name and model name
defaultname = "FVV"
default_wandb_project = "FVV"

### Warnings supression
warnings.filterwarnings("ignore", ".*Mean of empty slice.*")
warnings.filterwarnings("ignore", ".*invalid value encountered in scalar divide.*")
warnings.filterwarnings("ignore", ".*The given NumPy array is not writable.*")

### Data load
# Load dimensions of dataset from a saved file
with open(os.path.join("preprocessed_data","dimensions.txt"), "r") as file:
    z_dim = int(file.read())
# Load preprocessed data from memory-mapped files
xdat_disk = np.memmap(os.path.join("preprocessed_data","xdat.memmap"), 
                      dtype=np.float16, mode="r", shape=(z_dim,2,256,256)) # b1000 b0 ADC FLAIR
ydat_disk = np.memmap(os.path.join("preprocessed_data","ydat.memmap"), 
                      dtype=np.uint8, mode="r", shape=(z_dim,2,256,256)) # blob blob_viz
maskdat_disk = np.memmap(os.path.join("preprocessed_data","mask.memmap"), 
                         dtype=np.uint8, mode="r", shape=(z_dim,1,256,256)) # brainmask
metadat_disk = np.memmap(os.path.join("preprocessed_data","meta.memmap"), 
                         dtype=np.int32, mode="r", shape=(z_dim,5))
meta_valid_index = 4
pat_ids = pd.read_csv(os.path.join("preprocessed_data","subjects.csv"), index_col=0)

def main_train_func(config, run):
    global xdat_disk, ydat_disk, maskdat_disk, metadat_disk, pat_ids  
    if run is not None: plt.switch_backend("agg")
    print("KFOLD", config["kfold"])
    # Create a filter to exclude test data belonging to the current k-fold
    filter_disk = metadat_disk[:,3] != int(config["kfold"])

    # Preload data into RAM if specified in the configuration
    if config["preload_RAM"]:
        print("RAM usage: ", psutil.virtual_memory()[3]/(1024*1000*1000))
        print("loading X in ram")
        xdat = xdat_disk[filter_disk]
        del xdat_disk
        print("loading Y in ram")
        ydat = ydat_disk[filter_disk]
        del ydat_disk
        print("loading mask in ram")
        maskdat = maskdat_disk[filter_disk]
        del maskdat_disk
        print("loading meta in ram")
        metadat = metadat_disk[filter_disk]
        del metadat_disk
        print("loaded in ram")
        print("RAM usage: ", psutil.virtual_memory()[3]/(1024*1000*1000))
        
        train_filter = np.array(metadat[:,meta_valid_index] == 0)
        valid_filter = ~train_filter
    else:
        # If not preloading into RAM, use live memory-mapped data
        xdat = xdat_disk
        ydat = ydat_disk
        maskdat = maskdat_disk
        metadat = metadat_disk
        train_filter = np.logical_and(filter_disk, metadat[:,meta_valid_index] == 0)
        valid_filter = np.logical_and(filter_disk, metadat[:,meta_valid_index] == 1)

    
    # Initialize the TrainingObject with configuration and data selectors
    status = TrainingObject(
        config = config,
        run = run,
        xselector = tuple([0,1]), # dimensions for input data (b1000, b0)
        yselector = tuple([0,1]), # dimensions for output data (stroke, flairviz)
        blob_index_ydat = 0, # index for stroke outline in ydat
        viz_index_ydat = 1, # index for flairviz index in ydat
        mask_index = 0, # index for mask outline in model output
        blob_index = 1, # index for stroke outline in model output
        viz_index = 2, # index for flairviz index in model output
        # selector for z-slices
        zselector = list(range(-int((config["input_2.5_zslices"]-1)/2), +int((config["input_2.5_zslices"]-1)/2)+1)),
        epoch = config["start_epoch"],
        pat_ids=pat_ids
    )
    print("Selectors", status.xselector, status.cselector, status.yselector, status.zselector, status.zcenter)
    
    if config["save_model"] and run is not None:
        os.makedirs("saved_models", exist_ok=True)
        with open(os.path.join("saved_models",run.name+".config.json"), "w") as f:
            f.write(status.to_json())

    #### Model Creation and device assignment
    status.model, status.device = create_FVV_model(status)
    print("Model loaded")
    
    ##### DEFINE DATALOADERS
    # Create datasets and data loaders for training and validation
    whole_dataset_augm = FVVDataset(xdat,maskdat,metadat,ydat,status=status)
    whole_dataset_augm.transform = True
    whole_dataset = FVVDataset(xdat,maskdat,metadat,ydat,status=status)
    whole_dataset.transform = False
    train_dataset = Subset(whole_dataset_augm, np.where(train_filter)[0])
    valid_dataset = Subset(whole_dataset, np.where(valid_filter)[0])
    status.train_dataloader = DataLoader(train_dataset, batch_size=config["train_batch_size"], shuffle=True)
    status.valid_dataloader = DataLoader(valid_dataset, batch_size=config["test_batch_size"])
    bn_dataset = Subset(whole_dataset, np.where(train_filter)[0])
    status.bn_dataloader = DataLoader(bn_dataset, batch_size=16, shuffle=True)
    print("Dataloaders created")

    ##### Selection of subjects for data visualisation during validation
    simple_index_viz = []
    simple_index_noviz = []
    simple_index_mixviz = []
    validpat = list(set(metadat[valid_filter][:,0]))
    n = 0
    for ipat in range(len(validpat)):
        slices = metadat[:,0]==validpat[ipat]
        maxblob = np.argmax(np.sum(ydat[slices,status.blob_index_ydat], axis=(1,2)))
        if np.sum(ydat[slices][maxblob,status.blob_index_ydat]) == 0: ## avoid subjects with no stroke
            continue
        ratioviz = np.sum(ydat[slices][maxblob,status.viz_index_ydat])/np.sum(ydat[slices][maxblob,status.blob_index_ydat])
        if ratioviz > 0.95 and len(simple_index_viz) < 8:
            simple_index_viz.append(np.arange(metadat.shape[0])[slices][maxblob])
        elif ratioviz < 0.05 and len(simple_index_noviz) < 8:
            simple_index_noviz.append(np.arange(metadat.shape[0])[slices][maxblob])
        elif ratioviz > 0.20 and ratioviz < 0.80 and len(simple_index_mixviz) < 8:
            simple_index_mixviz.append(np.arange(metadat.shape[0])[slices][maxblob])
        if len(simple_index_viz) >= 8 and len(simple_index_noviz) >= 8 and len(simple_index_mixviz) >= 8:
            break
    # Create data loaders for visualization
    simple_viz_dataset = Subset(whole_dataset, simple_index_viz)
    status.simple_viz = DataLoader(simple_viz_dataset, batch_size=8) # create your dataloader
    simple_noviz_dataset = Subset(whole_dataset, simple_index_noviz)
    status.simple_noviz = DataLoader(simple_noviz_dataset, batch_size=8) # create your dataloader
    simple_mixviz_dataset = Subset(whole_dataset, simple_index_mixviz)
    status.simple_mixviz = DataLoader(simple_mixviz_dataset, batch_size=8) # create your dataloader
    print("Figures subjects loaded", len(simple_index_viz), len(simple_index_noviz), len(simple_index_mixviz))

    # Loss functions
    status.bce = smp.losses.SoftBCEWithLogitsLoss(smooth_factor=config["label_smoothing"],
                                           reduction="none")
    status.bce_weighted = smp.losses.SoftBCEWithLogitsLoss(smooth_factor=config["label_smoothing"],
                                           reduction="none",
                                           pos_weight=torch.FloatTensor([config["weight_viz"]]).to(device) if config["weight_viz"] else None)
    status.acc = BinaryAccuracy(threshold=0.0)

    # optimizer
    if config["optimiser_name"] == "SGD":
        status.optimizer = torch.optim.SGD(status.model.parameters(), lr=config["learning_rate"])
    elif config["optimiser_name"] == "Adam":
        status.optimizer = torch.optim.Adam(status.model.parameters(), lr=config["learning_rate"])

    # Learning rate schedulers
    if config["lr_scheduler"]:
        if config["lr_scheduler_mode"] == "plateau":
            status.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                status.optimizer,
                mode='min', 
                factor=config["lr_scheduler_factor"],
                patience=config["lr_scheduler_patience"],
                threshold=config["lr_scheduler_threshold"],
                threshold_mode='abs',
                cooldown=config["lr_scheduler_cooldown"],
                min_lr=config["lr_scheduler_min_lr"])
        if config["lr_scheduler_mode"] == "cosine_warm_restart":
            status.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                status.optimizer,
                T_0=config["lr_scheduler_patience"], 
                eta_min=config["lr_scheduler_min_lr"])
            
    print("Optimizer loaded")

    # Create initial figures if in debug mode
    if run is None:
        create_figure(status.simple_viz, status)
        create_figure(status.simple_noviz, status)
        create_figure(status.simple_mixviz, status)
        plt.show()
        
    # Initialize best metrics
    best_validation_loss = 10000.0
    best_global_dice = -1.0
    best_kappa = -1.0
    best_auc = -1.0

    # MAIN LOOP
    for epoch in range(status.epoch,config["max_num_epochs"]+1):  # loop over the dataset multiple times    
        status.epoch = epoch
        ### VALIDATION LOOP
        if epoch % config["validation_each_epoch"] == 0 and epoch > 0:
            bn_loop(status) # data normalization before validation to reset BN layers
            this_loss, this_dice, this_kappa, this_auc = validation_loop(status)
            # Save the model if it is an improvement
            if config["save_model"] and epoch > 0:
                willsave = False
                if this_loss < best_validation_loss or this_dice > best_global_dice or \
                    this_kappa > best_kappa or this_auc > best_auc:
                    if run is not None:
                        if status.config["multigpu"]: state_dict = status.model.module.state_dict()
                        else: state_dict = status.model.state_dict()
                        torch.save(state_dict, os.path.join("saved_models",status.run.name+"-epoch"+str(status.epoch)+".pth"))
                best_validation_loss = min(best_validation_loss,this_loss)
                best_global_dice = max(best_global_dice,this_dice)
                best_kappa = max(this_kappa,best_kappa)
                best_auc = max(this_auc,best_kappa)
                
        ### BREAK STEP
        if epoch == config["max_num_epochs"]:
            if run is not None: wandb.finish()
            if config["save_model"]:
                if status.config["multigpu"]: state_dict = status.model.module.state_dict()
                else: state_dict = status.model.state_dict()
                torch.save(state_dict, os.path.join("saved_models",status.run.name+"-epoch"+str(status.epoch)+".pth"))
            break
            
        ### TRAINING STEP
        training_loop(status)

    # Save final model
    if status.config["multigpu"]: state_dict = status.model.module.state_dict()
    else: state_dict = status.model.state_dict()
    torch.save(state_dict, os.path.join("saved_models",status.run.name+"-epoch"+str(status.epoch)+".pth"))
    if run is not None: wandb.finish()

if __name__ == "__main__":    
    # Argument parser for running the script from command line
    parser = argparse.ArgumentParser(description="DeepLearner",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-k", "--kfold", help="fold", default=0.0)
    parser.add_argument("-n", "--name", help="name", default=defaultname)
    parser.add_argument("-d", "--debug", help="debug", default=None)
    parser.add_argument("-p", "--preload_model", help="Preload Model", default=None)
    parser.add_argument("-ram", "--preload_RAM", help="preload_RAM", default=None)
    parser.add_argument("-e", "--num_epochs", help="Max number of epochs", default=75)
    parser.add_argument("-lr", "--learning_rate", help="Initial learning rate", default=0.001)
    parser.add_argument("-w", "--wandb_project", help="wandb project name", default=default_wandb_project)
    args = parser.parse_args()
    
    # Configuration settings for the training process
    config = {
        "train_batch_size": 16,
        "test_batch_size": 128,
        "max_num_epochs": int(vars(args)["num_epochs"]), 
        "slices_per_epoch": 128*128,
        "label_smoothing": 0.01,
        "lambda_mask": 1,
        "lambda_coarse_segm": 1,
        "lambda_viz": 100,
        "lambda_noviz": 0,
        "weight_viz": 0,
        "learning_rate": float(vars(args)["learning_rate"]), 
        "lr_scheduler":True,
        "lr_scheduler_mode":"plateau",
        "lr_scheduler_factor":0.1,
        "lr_scheduler_patience":6,
        "lr_scheduler_threshold":0.1,
        "lr_scheduler_cooldown":2,
        "lr_scheduler_min_lr":0.000001,
        "start_epoch":0,
        "validation_each_epoch": 1,
        "augmentation": True,
        "aug_brightness":0.3,
        "aug_contrast":0.3,
        "aug_angle":8,
        "aug_translate":0.04,
        "aug_scale":1.1,
        "aug_shear":8,
        "aug_z":True,
        "input_2.5D": "smp3d",
        "input_2.5_zslices": 7,
        "preload_model": vars(args)["preload_model"], 
        "model_name": "DeepLabV3+",
        "encoder_name": "efficientnet-b0",
        "encoder_weights": "imagenet",
        "optimiser_name": "Adam",
        "save_model": True,
        "multigpu": True,
        "debug": True if vars(args)["debug"] else False,
        "kfold": vars(args)["kfold"],
        "preload_RAM": True if vars(args)["preload_RAM"] else False,
        "wandb_project": vars(args)["wandb_project"]
    }

    # Run the training function in debug mode or normal mode
    if config["debug"]:
        main_train_func(config, None)
    else:
        run = wandb.init(project=config["wandb_project"], config=config, name=vars(args)["name"]+"-k"+str(int(config["kfold"])))
        main_train_func(config, run)
        wandb.finish()