######### Script for Data Preparation #########
# J. Benzakoun 2024
#
# Usage:
#   python 1-DataPreparation.py
#
# This script prepares data for deep learning training by processing MRI images,
# creating memory-mapped files for efficient data handling, and generating metadata
# for each slice. The input data is expected to be organized in a specific directory
# structure as described below.
#
# Expected data input:
#   input_data/
#    |
#    |---SubjectXYZ/
#    |      |
#    |      |---b0.nii.gz
#    |      |---b1000.nii.gz
#    |      |---mask_roi.nii.gz : optional, will be computed if needed
#    |      |---stroke_roi.nii.gz : large binary ROI over the stroke area
#    |      |---flairviz_roi.nii.gz : regional delineation of FLAIR visible areas
#    |
#    |---SubjectABC/
#    |      |
#    |      |--- ...
######################################################################

import glob
import os
import nibabel as nib
from tqdm import tqdm
import numpy as np
from lib.tools import nib2rescaled
from sklearn.model_selection import KFold, train_test_split
import pandas as pd

### Default settings
# Number of k-folds for cross-validation
num_kfolds = 10
# Proportion of data to be used for validation
valid_proportion = 0.2

# Get the list of patient directories containing the "b1000.nii.gz" file
list_pats = sorted(glob.glob(os.path.join("input_data","*","b1000.nii.gz")))
list_pats = [os.path.basename(os.path.dirname(i)) for i in list_pats]

## FAST scrolling to get total number of slices
print("Calculation of the number of slices")
zcount = 0
for pat in tqdm(list_pats):
    # Load the "b1000.nii.gz" file for each patient and accumulate the number of slices
    zcount += int(nib.load(os.path.join("input_data",pat,"b1000.nii.gz")).shape[2])
print("Total number of slices : ", zcount)

## Creation of memmap files for storing data
# Create output directory if it does not exist
os.makedirs("preprocessed_data", exist_ok=True)
# Create memory-mapped files for data storage
os.makedirs("preprocessed_data", exist_ok=True)
xdat = np.memmap(os.path.join("preprocessed_data","xdat.memmap"), 
                 dtype=np.float16, mode="w+", shape=(zcount,2,256,256)) # b1000 b0
ydat = np.memmap(os.path.join("preprocessed_data","ydat.memmap"),  
                 dtype=np.uint8, mode="w+", shape=(zcount,2,256,256)) # stroke_roi, flairviz_roi
maskdat = np.memmap(os.path.join("preprocessed_data","mask.memmap"),  
                    dtype=np.uint8, mode="w+", shape=(zcount,1,256,256)) # mask_roi
metadat = np.memmap(os.path.join("preprocessed_data","meta.memmap"), 
                    dtype=np.int32, mode="w+", shape=(zcount,5))
            # ID, sliceid, is_stroke_slice, kfold_group, validation subject

# Store the total number of slices in a text file
with open(os.path.join("preprocessed_data","dimensions.txt"), "w") as file:
    file.write(str(zcount))

## Definition of k-folds and validation subjects
# Create k-fold splits for the list of patients
skf = KFold(n_splits=num_kfolds, shuffle=True, random_state=2000)
kfold_group = [-1]*len(list_pats)
# Assign each patient to a k-fold group
for i, (train_index, test_index) in enumerate(skf.split(np.arange(len(list_pats)))):
    for k in test_index:
        kfold_group[k] = i

# Define training and validation subjects for each fold
train_valid = [0]*len(list_pats)
for fold in range(num_kfolds):
    indices_in_fold = [i for i in range(len(list_pats)) if kfold_group[i] == fold]
    _, valid_index = train_test_split(indices_in_fold, test_size=valid_proportion)
    for k in valid_index:
        train_valid[k] = 1

## Main loop for data creation
z_i = 0
synthesis_array = []
# Loop through each patient to process their data
for index, pat in tqdm(enumerate(list_pats), total=len(list_pats)):
    ### Checking if mask exists
    maskpath = None
    if os.path.exists(os.path.join("input_data",pat,"mask_roi.nii.gz")):
        maskpath = os.path.join("input_data",pat,"mask_roi.nii.gz")
    # Load the "b1000.nii.gz" file
    b1000nib = nib.load(os.path.join("input_data",pat,"b1000.nii.gz"))
                        
    ### calling rescaling function to prepare the data
    zlen, volvox, scaling, norm_b0, norm_b1000, rescaled_mask, rescaled_stroke_roi, rescaled_flairviz_roi = nib2rescaled(
        os.path.join("input_data",pat,"b0.nii.gz"),
        os.path.join("input_data",pat,"b1000.nii.gz"),
        maskpath,
        os.path.join("input_data",pat,"stroke_roi.nii.gz"),
        os.path.join("input_data",pat,"flairviz_roi.nii.gz")
    )
    

    ### Construction of memory-mapped data
    # Store rescaled data into memmap files
    xdat[z_i:z_i+zlen,0,:,:] = norm_b1000.transpose((2,0,1))
    xdat[z_i:z_i+zlen,1,:,:] = norm_b0.transpose((2,0,1))
    maskdat[z_i:z_i+zlen,0,:,:] = rescaled_mask.transpose((2,0,1))
    ydat[z_i:z_i+zlen,0,:,:] = rescaled_stroke_roi.astype(np.uint8).transpose((2,0,1))
    ydat[z_i:z_i+zlen,1,:,:] = rescaled_flairviz_roi.astype(np.uint8).transpose((2,0,1))
    # Store metadata for each slice
    metadat[z_i:z_i+zlen,0] = index # patient ID
    metadat[z_i:z_i+zlen,1] = np.arange(zlen) # slice ID
    metadat[z_i:z_i+zlen,2] = rescaled_stroke_roi.sum(axis=(0,1))>0 # Indicate if the slice contains stroke
    metadat[z_i:z_i+zlen,3] = kfold_group[index] # Assign k-fold group
    metadat[z_i:z_i+zlen,4] = train_valid[index] # Indicate if the subject is for validation

    synthesis_array.append({"Patient":pat,"ID":index,"Slices number":zlen,"KFold":kfold_group[index],
                            "volvox":volvox,"scaling":scaling,"b1000X":b1000nib.shape[0],"b1000Y":b1000nib.shape[1]})
    z_i += zlen

### Save the dataset characteristics to a CSV file for later analysis
pd.DataFrame(synthesis_array).to_csv(os.path.join("preprocessed_data","subjects.csv"))