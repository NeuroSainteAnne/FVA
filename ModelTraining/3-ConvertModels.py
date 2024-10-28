###############
# Converts selected model to ONNX format
# J. Benzakoun 2024
#
# Usage:
#   python 3-ConvertModels.py
#
# This script converts selected PyTorch models to ONNX format for easier deployment and compatibility.
# If no models are manually selected in script header, the script will automatically choose the latest model for each k-fold.
#
######################################################################

# List of models to be converted. If empty, the script will automatically select the latest model for each fold
list_models = [] # We strongly advise you to manually select your models! If you do not select them, the latest model will be selected for each fold

# module importation
import torch
import os
import glob
import numpy as np
import re
import sys
sys.path.insert(0,os.path.join(os.getcwd(),"../dependencies/smp3d"))
from modules.classes import TrainingObject
from modules.model import create_FVV_model

# Function to extract the epoch number from the model filename for sorting purposes
def natural_keys(text):
    return int(text.split('epoch')[1].replace(".pth",""))

# If no models are provided, automatically select the latest model for each k-fold
if len(list_models) == 0:
    # Load dataset dimensions from a saved file
    with open(os.path.join("preprocessed_data","dimensions.txt"), "r") as file:
        z_dim = int(file.read())
        
    # Load metadata from memory-mapped file
    metadat = np.memmap(os.path.join("preprocessed_data","meta.memmap"), 
                         dtype=np.int32, mode="r", shape=(z_dim,5))
    # Determine the number of k-folds based on metadata
    kfolds = np.max(metadat[:,3])+1
    
    # For each k-fold, select the latest available model
    for k in range(kfolds):
        available_models = glob.glob(os.path.join("saved_models","*-k"+str(k)+"-epoch*.pth"))
        available_models.sort(key=natural_keys)  # Sort models by epoch number
        available_models.reverse()  # Reverse to get the latest model first
        
        # Append the latest model to the list of selected models
        if len(available_models) > 0:
            list_models.append(available_models[0])
        else:
            print("Error: did not found model for k-fold", k)
            exit()

# Create the output directory for ONNX models if it doesn't exist
os.makedirs("saved_models_ONNX", exist_ok=True)

# Loop through each selected model and convert it to ONNX format
for i, m in enumerate(list_models):
    previous_status_path = m.split("-epoch")[0]+".config.json"
    if os.path.exists(previous_status_path):
        with open(previous_status_path, "r") as pv: 
            previous_status = TrainingObject.from_json(pv.read())
    else:
        print("Did not find config file")
        continue

    # Load the PyTorch model
    previous_status.config["preload_model"] = m
    previous_status.config["multigpu"] = False
    model, device = create_FVV_model(previous_status, device="cpu")
        
    model.eval()   # Set the model to evaluation mode
    
    # Create a dummy input tensor with the expected dimensions
    torch_input = torch.randn(1, 2, 7, 256, 256, requires_grad=True).cpu()
    
    # Export the model to ONNX format
    torch.onnx.export(model,         # model being run 
             torch_input,       # model input (or a tuple for multiple inputs) 
             os.path.join("saved_models_ONNX","model"+str(i+1)+".onnx"),       # where to save the model  
             export_params=True,  # store the trained parameter weights inside the model file 
             do_constant_folding=True,  # whether to execute constant folding for optimization 
             input_names = ['modelInput'],   # the model's input names 
             output_names = ['modelOutput'], # the model's output names 
             dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                           'modelOutput' : {0 : 'batch_size'}}) 