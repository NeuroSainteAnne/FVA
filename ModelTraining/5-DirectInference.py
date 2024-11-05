######### Script for Direct Inference (external validation) #########
# J. Benzakoun 2024
#
# Usage:
#   python 5-DirectInference.py  [-d inference_dir] 
#
# Arguments:
#   -d, --inference_dir   : Folder on which the inference will be performed
#   -f, --favorite        : GPU or CPU (default: GPU)
#
# Expected data input:
#   inference_data/
#    |
#    |---SubjectXYZ/
#    |      |
#    |      |---b0.nii.gz
#    |      |---b1000.nii.gz
#    |
#    |---SubjectABC/
#    |      |
#    |      |--- ...
######################################################################

import argparse
import os
import glob
from modules.tools import nib2rescaled, quick_zstack, getLargestCC, autorescale, compute_color_nib
from tqdm import tqdm
import nibabel as nib
import re
import numpy as np
import onnxruntime
import pandas as pd

list_pth_models = []
type_load = "onnx" # autodetect if onnx

# Argument parser for running the script from the command line
parser = argparse.ArgumentParser(description="DataPrepare",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--inference_dir", help="Inference directory", default="input_data/")
parser.add_argument("-f", "--favorite", help="favorite inference device", default="GPU")
inference_dir = str(vars(parser.parse_args())["inference_dir"])
favorite_device = str(vars(parser.parse_args())["favorite"])
if not os.path.exists(inference_dir):
    print("ERROR: did not find", inference_dir, "directory")
    exit()

# Get the list of patient directories containing the "b1000.nii.gz" file
list_pats = sorted(glob.glob(os.path.join("input_data","*","b1000.nii.gz")))
list_pats = [os.path.basename(os.path.dirname(i)) for i in list_pats]

##### KFolds
# Function to extract numerical keys for sorting models based on epoch numbers
def natural_keys(text):
    return int(re.split(r'(\d+)', text)[1])
# List all ONNX models for k-fold inference
if type_load == "pth":
    import sys
    sys.path.insert(0,os.path.join(os.getcwd(),"../dependencies/smp3d"))
    from modules.model import create_FVV_model
    from modules.classes import TrainingObject
    import torch
    list_models = list_pth_models
else:
    list_models = glob.glob(os.path.join("saved_models_ONNX","model*.onnx"))
    list_models.sort(key=natural_keys)
num_kfolds = len(list_models)
float_k = float(num_kfolds)

metrics = []
for pat in tqdm(sorted(list_pats)):
    b1000nib = nib.load(os.path.join(inference_dir,pat,"b1000.nii.gz"))
                        
    ### calling rescaling function to prepare the data
    zlen, volvox, scaling, norm_b0, norm_b1000, rescaled_mask, _, _ = nib2rescaled(
        os.path.join(inference_dir,pat,"b0.nii.gz"),
        os.path.join(inference_dir,pat,"b1000.nii.gz")
    )

    stacked = quick_zstack(norm_b0, norm_b1000)

    # perform inference
    outputs = []
    for model in list_models:
        if type_load == "onnx":
            if favorite_device == "GPU":
                ort_session = onnxruntime.InferenceSession(model, providers=['CUDAExecutionProvider','CPUExecutionProvider'])
            else:
                ort_session = onnxruntime.InferenceSession(model, providers=['CPUExecutionProvider','CUDAExecutionProvider'])
            outputs.append(ort_session.run(None,{"modelInput":stacked.astype(np.float32)})[0])
        elif type_load == "pth":
            previous_status_path = model.split("-epoch")[0]+".config.json"
            with open(previous_status_path, "r") as pv: 
                previous_status = TrainingObject.from_json(pv.read())
            # Load the PyTorch model
            qat_trained = False
            if "qat_finetune" in previous_status.config.keys() and previous_status.config["qat_finetune"]:
                previous_status.config["preload_model"] = None
                previous_status.config["qat_preload_model"] = model
                qat_trained = True
            else:
                previous_status.config["preload_model"] = model
            previous_status.config["multigpu"] = False
            model, device = create_FVV_model(previous_status, device="cpu")
            model.eval()   # Set the model to evaluation mode
            if qat_trained:
                model = torch.ao.quantization.convert(model.cpu())
            outputs.append(model(torch.Tensor(stacked.astype(np.float32))))
    
    patdata = np.stack(outputs).transpose((3,4,1,2,0))
    if patdata.shape[3] == 4: patdata = patdata[:,:,:,(0,2,3)] # backward compatibility

    # Compute brain mask based on majority voting
    mask_raw = patdata[:,:,:,0] > 0
    mask_l = np.stack([getLargestCC(mask_raw[...,i]) for i in range(num_kfolds)], axis=-1)
    mask_l_majority = np.sum(mask_l, axis=-1)>=(float_k/2)
    brainmaskvox = np.sum(mask_l_majority)
    
    # Compute stroke mask based on majority voting
    coarse_raw = patdata[:,:,:,1] > 0
    coarse_l = coarse_raw*mask_raw
    coarse_l_majority = (mask_l_majority*np.sum(coarse_raw, axis=-1))>=(float_k/2)
    blobmaskvox = np.sum(coarse_l_majority)
    
    # Compute sum of FVV predictions
    viz_raw = patdata[:,:,:,2] > 0
    viz_l_stroke = [viz_raw[coarse_l_majority>0.5,i] for i in range(num_kfolds)]
    viz_lin_stroke = [i.sum() for i in viz_l_stroke]
    viz_stroke = coarse_l_majority*np.sum(viz_raw, axis=-1).astype(float)
    vizmaskvox = np.mean(viz_lin_stroke)
    
    # Rescale mask, stroke, and FVV predictions
    invscale = 1/scaling
    mask_pred, _ = autorescale(mask_l_majority.astype(bool), 
                               force_scaling=invscale, targetX=b1000nib.shape[0], targetY=b1000nib.shape[1])
    stroke_pred, _ = autorescale(coarse_l_majority.astype(bool), 
                               force_scaling=invscale, targetX=b1000nib.shape[0], targetY=b1000nib.shape[1])
    viz_pred, _ = autorescale(viz_stroke, 
                               force_scaling=invscale, targetX=b1000nib.shape[0], targetY=b1000nib.shape[1])

    # Save raw predictions
    nib.save(nib.Nifti1Image(np.stack([mask_pred*num_kfolds,
                                       stroke_pred*num_kfolds,
                                       viz_pred],axis=3).astype(np.uint8), b1000nib.affine),
             os.path.join(inference_dir,pat,"predictions.nii.gz"))

    # Save rgb nib
    colornib = compute_color_nib(b1000nib, mask_pred, stroke_pred, viz_pred, maxvizval=float_k)
    nib.save(colornib, os.path.join(inference_dir,pat,"heatmap.nii.gz"))

    # Concatenate data
    patsynth = {"Patient":pat,
                "BrainMaskVoxels":brainmaskvox,
                "StrokeBlobVoxels":blobmaskvox,
                "FlairVizVoxels":vizmaskvox,
                "BrainMaskVolume":brainmaskvox*volvox,
                "StrokeBlobVolume":blobmaskvox*volvox,
                "FlairVizVolume":vizmaskvox*volvox,
                "VoxVolume after rescaling":volvox}
    with open(os.path.join(inference_dir,pat,"results.txt"), "w") as synfile:
        synfile.write("\n".join([k+"="+str(v) for k, v in patsynth.items()]))
    metrics.append(patsynth)
    
pd.DataFrame(metrics).to_csv(os.path.join(inference_dir,"results.csv"))