###############
# Inference Test Script
# J. Benzakoun 2024
#
# This script performs inference using ONNX models on MRI data. It handles loading the dataset,
# performing predictions using pre-trained models, and optionally generating heatmaps for visualization.
######################################################################

# Configuration parameters
batch_size = 32
input_zslices = 7
overwrite_inference_memmap = True
overwrite_inference_csv = True
generate_heatmap = True

# modules loading
import os
import glob
from lib.tools import autorescale, getLargestCC, compute_color_nib
from lib.generator import FVVDataset
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
import re
import numpy as np
import pandas as pd
import onnxruntime
import nibabel as nib
import warnings

##### KFolds
# Function to extract numerical keys for sorting models based on epoch numbers
def natural_keys(text):
    return int(re.split(r'(\d+)', text)[1])
# List all ONNX models for k-fold inference
list_models = glob.glob(os.path.join("saved_models_ONNX","model*.onnx"))
list_models.sort(key=natural_keys)
num_kfolds = len(list_models)

##### Define selectors
xselector = tuple([0,1]) # dimensions for input data (b1000, b0)
yselector = tuple([0,1]) # dimensions for output data (stroke, flairviz)
mask_index = 0 # index for mask prediction 
blob_index = 1 # index for stroke prediction 
viz_index = 2 # index for flairviz prediction 
# Define the range of z-slices for input
zselector = list(range(-int((input_zslices-1)/2), +int((input_zslices-1)/2)+1))
cselector = len(xselector)
zcenter = np.where(np.array(zselector)==0)[0][0] 
out_dim = 3 # Number of output channels

### Warnings supression
warnings.filterwarnings("ignore", ".*The given NumPy array is not writable.*")
warnings.filterwarnings("ignore", ".*Mean of empty slice.*")
warnings.filterwarnings("ignore", ".*invalid value encountered in scalar divide.*")

##### Load dataset
# Load memmap dimensions
with open(os.path.join("preprocessed_data","dimensions.txt"), "r") as file:
    z_dim = int(file.read())
    
# Load metadata from memory-mapped file
metadat = np.memmap(os.path.join("preprocessed_data","meta.memmap"), 
                     dtype=np.int32, mode="r", shape=(z_dim,5))

# Load patient data csv
pat_ids = pd.read_csv(os.path.join("preprocessed_data","subjects.csv"), index_col=0)
    
# Check if previous inference memmap exists and load it, or create a new one
if os.path.exists(os.path.join("preprocessed_data","inference.memmap")) and not overwrite_inference_memmap:
    print("Preloading previous inference memmap")
    out_pred = np.memmap(os.path.join("preprocessed_data","inference.memmap"), dtype=np.uint8, mode="r", shape=(z_dim,out_dim,256,256))
else:
    print("INFERENCE process")
    # Load input data and mask data from memory-mapped files
    xdat = np.memmap(os.path.join("preprocessed_data","xdat.memmap"), dtype=np.float16, mode="r", shape=(z_dim,2,256,256)) # b1000 b0
    maskdat = np.memmap(os.path.join("preprocessed_data","mask.memmap"), dtype=np.uint8, mode="r", shape=(z_dim,1,256,256)) # brainmask
    # Create a new memory-mapped file for storing predictions
    out_pred = np.memmap(os.path.join("preprocessed_data","inference.memmap"), dtype=np.uint8, mode="w+", shape=(z_dim,out_dim,256,256))
    # Append original slice index to metadata
    metadat_with_sliceid = np.concatenate([metadat, np.arange(metadat.shape[0])[:,np.newaxis]], axis=1)
    
    # Perform inference for each model (k-fold)
    for model in list_models:
        print("Model", model)
        kfold = int(os.path.basename(model).replace("model","").replace(".onnx",""))-1
        # load ONNX model
        ort_session = onnxruntime.InferenceSession(model, 
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        
        # Create dataset and dataloader for inference
        whole_dataset = FVVDataset(xdat,maskdat,metadat_with_sliceid,input25D=True,xselector=xselector,zselector=zselector)
        test_filter = metadat_with_sliceid[:,3] == kfold ## select test kfold
        test_dataset = Subset(whole_dataset, np.where(test_filter)[0])
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Perform inference for each batch
        with tqdm(total=len(test_dataloader)) as pbar:
            for i, data in enumerate(test_dataloader, 0):
                pbar.update(1)
                original_id = data[2][:,-1]
                outputs = ort_session.run(None,{"modelInput":data[0].float().detach().cpu().numpy()})[0]
                out_pred[original_id] = outputs > 0

# Compute metrics if CSV file does not exist or overwrite is enabled
if os.path.exists(os.path.join("processed_data","test_inference.csv")) and not overwrite_inference_csv:
    print("Avoiding metrics computation")
else:
    print("Computing Metrics...")
    os.makedirs("processed_data", exist_ok=True)
    # Initialize metrics dictionary
    metrics={key:[] for key in ["Patient",
                                "mask_voxels","blob_voxels","viz_voxels_in_mask",
                                "vol_vox","viz_voxels_in_stroke",
                                "mask_volume","blob_volume","viz_volume_in_mask",
                                "viz_volume_in_stroke",
                                "meanviz_in_blob","meanviz_in_stroke"]}
    all_filter = metadat[:,1]==0 # first slice al each patient
    all_indices =  metadat[all_filter,0].astype(np.uint32) # all patient indices
    
    # Loop through each patient to compute metrics
    for ID in tqdm(all_indices):
        filtpat = metadat[:,0]==ID
        patdata = out_pred[filtpat]
        
        volvox = float(pat_ids.iloc[int(ID)]["volvox"])
        pat_name = pat_ids.iloc[int(ID)]["Patient"]
        metrics["Patient"] += [pat_name]
        
        mask_raw = patdata[:,mask_index]
        mask_l = getLargestCC(mask_raw>0.5)
        metrics["mask_voxels"] += [np.sum(mask_l)]
        
        coarse_raw = patdata[:,blob_index]
        coarse_l = mask_l*(coarse_raw>0.5)
        metrics["blob_voxels"] += [np.sum(coarse_l)]
        
        viz_raw = patdata[:,viz_index]
        viz_l_stroke = viz_raw[coarse_l]
        metrics["meanviz_in_stroke"] += [np.mean(viz_l_stroke)]
        
        viz_l_mask = viz_raw[mask_l]
        metrics["meanviz_in_blob"] += [np.mean(viz_l_mask)]
        
        viz_thr_mask = np.sum(viz_l_mask>0.5)
        metrics["viz_voxels_in_mask"] += [viz_thr_mask]
        viz_thr_stroke = np.sum(viz_l_stroke>0.5)
        metrics["viz_voxels_in_stroke"] += [viz_thr_stroke]
        
        metrics["vol_vox"] += [volvox]
        
    # Compute volume metrics from voxel counts
    metrics["mask_volume"] = np.array(metrics["mask_voxels"])*np.array(metrics["vol_vox"])
    metrics["blob_volume"] = np.array(metrics["blob_voxels"])*np.array(metrics["vol_vox"])
    metrics["viz_volume_in_mask"] = np.array(metrics["viz_voxels_in_mask"])*np.array(metrics["vol_vox"])
    metrics["viz_volume_in_stroke"] = np.array(metrics["viz_voxels_in_stroke"])*np.array(metrics["vol_vox"])
    
    # Save metrics to CSV
    pd.DataFrame(metrics, index=None).to_csv(os.path.join("processed_data","test_inference.csv"), index=False)

# Generate heatmaps for each patient if enabled
if generate_heatmap:
    print("Generating heatmaps...")
    for _, patline in tqdm(pat_ids.iterrows(), total=len(pat_ids)):
        patselector = metadat[:,0] == patline["ID"]
        patpred = out_pred[patselector]
        b1000nib = nib.load(os.path.join("input_data",patline["Patient"],"b1000.nii.gz"))
        invscale = float(1/patline["scaling"])
        
        # Rescale and compute heatmap for mask, stroke, and flairviz predictions
        mask_pred, _ = autorescale(patpred[:,mask_index].transpose((1,2,0)).astype(bool), force_scaling=invscale, targetX=patline["b1000X"], targetY=patline["b1000Y"])
        stroke_pred, _ = autorescale(patpred[:,blob_index].transpose((1,2,0)).astype(bool), force_scaling=invscale, targetX=patline["b1000X"], targetY=patline["b1000Y"])
        stroke_pred[~mask_pred] = False
        viz_pred, _ = autorescale(patpred[:,viz_index].transpose((1,2,0)).astype(bool), force_scaling=invscale, targetX=patline["b1000X"], targetY=patline["b1000Y"])
        viz_pred[~stroke_pred] = False
        
        # Generate color heatmap and save
        colornib = compute_color_nib(b1000nib, mask_pred, stroke_pred, viz_pred)
        os.makedirs(os.path.join("processed_data",patline["Patient"]), exist_ok=True)
        nib.save(colornib, os.path.join("processed_data",patline["Patient"],"heatmap.nii.gz"))