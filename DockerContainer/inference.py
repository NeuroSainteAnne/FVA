from datetime import datetime
t_first = datetime.now()

import glob
import os
from pathlib import Path
os.environ['MPLCONFIGDIR'] = os.path.join(Path.home(),"MPLCONFIG")
import torch
import onnxruntime
import scipy
import nibabel as nib
import numpy as np
from dipy.segment.mask import median_otsu
from scipy.ndimage import binary_fill_holes, label, binary_dilation, generate_binary_structure, binary_erosion
from skimage.transform import resize, rescale
from skimage import filters, morphology
from skimage.measure import label as sklabel
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import math 

from tools import autorescale, nib2rescaled, quick_zstack, compute_color_nib, getLargestCC
   
default_device = "cpu" # cuda or cpu
inference_mode = "ONNX" # NB quantized models cannot be run on GPU

if inference_mode == "PTH":
    import sys
    sys.path.insert(0,os.path.join(os.getcwd(),"modules"))
    from modules.model import create_FVV_model, QATOverhead
    from modules.classes import TrainingObject
    import segmentation_models_pytorch
    import segmentation_models_pytorch_3d
    previous_status_path = "PTH_models/model.config.json"
    with open(previous_status_path, "r") as pv: 
        previous_status = TrainingObject.from_json(pv.read())
    previous_status.config["preload_model"] = False
    previous_status.config["encoder_weights"] = None
    previous_status.config["multigpu"] = False
    model, device = create_FVV_model(previous_status, device=default_device)
    qat_trained = False
    if "qat_finetune" in previous_status.config.keys() and previous_status.config["qat_finetune"]:
        model.eval()
        model = torch.ao.quantization.convert(model.cpu())
        qat_trained = True
    print("created model")

t_loaded_modules = datetime.now()

print("CUDA",torch.cuda.is_available())

print("Scanning directory")
b0_potential = glob.glob("/app/data/*b0.nii.gz") + \
                glob.glob("/app/data/*b0.nii") + \
                glob.glob("/app/data/*B0.nii.gz") + \
                glob.glob("/app/data/*B0.nii")
b1000_potential = glob.glob("/app/data/*b1000.nii.gz") + \
                glob.glob("/app/data/*b1000.nii") + \
                glob.glob("/app/data/*B1000.nii.gz") + \
                glob.glob("/app/data/*B1000.nii") + \
                glob.glob("/app/data/*DWI.nii.gz") + \
                glob.glob("/app/data/*DWI.nii")
if len(b0_potential) > 0 and len(b1000_potential) > 0:
    b0path = b0_potential[0]
    b1000path = b1000_potential[0]
else:
    print("Did not found b0 and b1000 NIFTI files")
    exit() 
    
t_scanned_directory = datetime.now()

print("Data normalization")
b1000nib = nib.load(b1000path)

zlen, volvox, scaling, norm_b0, norm_b1000, rescaled_mask, _, _ = nib2rescaled(
    b0path,
    b1000path
)

t_data_normalized = datetime.now()

print("Data stacking")
    
x = quick_zstack(norm_b0,norm_b1000)

t_data_stacked = datetime.now()

print("Inference")
if inference_mode == "ONNX":
    onnxruntime_outputs = []
    for i in range(10):
        if default_device == "cpu":
            ort_session = onnxruntime.InferenceSession("./ONNX_models/model"+str(i+1)+".onnx", 
                providers=['CPUExecutionProvider'])
        else:
            ort_session = onnxruntime.InferenceSession("./ONNX_models/model"+str(i+1)+".onnx", 
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        onnxruntime_outputs.append(ort_session.run(None,{"modelInput":x.astype(np.float32)})[0])
else:
    onnxruntime_outputs = []
    for i in range(10):
        model.load_state_dict(torch.load("./PTH_models/model"+str(i+1)+".pth", weights_only=True))
        result_i = model(torch.Tensor(x.astype(np.float32)).to(device))
        onnxruntime_outputs.append(result_i)
        print("ended", i, result_i.shape)

t_inference_ended = datetime.now()

patdata = np.stack(onnxruntime_outputs).transpose((3,4,1,2,0))
if patdata.shape[3] == 4: patdata = patdata[:,:,:,(0,2,3)] # backward compatibility

mask_raw = patdata[:,:,:,0] > 0
mask_l = np.stack([getLargestCC(mask_raw[...,i]) for i in range(10)], axis=-1)
mask_l_majority = np.sum(mask_l, axis=-1)>=(10/2)
brainmaskvox = np.sum(mask_l_majority)

coarse_raw = patdata[:,:,:,1] > 0
coarse_l = coarse_raw*mask_raw
coarse_l_majority = (mask_l_majority*np.sum(coarse_raw, axis=-1))>=(10/2)
blobmaskvox = np.sum(coarse_l_majority)

viz_raw = patdata[:,:,:,2] > 0
viz_l_stroke = [viz_raw[coarse_l_majority>0.5,i] for i in range(10)]
viz_lin_stroke = [i.sum() for i in viz_l_stroke]
viz_stroke = coarse_l_majority*np.sum(viz_raw, axis=-1).astype(float)
vizmaskvox = np.mean(viz_lin_stroke)

outputs = [mask_l_majority.astype(bool), coarse_l_majority.astype(bool), viz_stroke]

outputs[0], _ = autorescale(outputs[0], force_scaling=1/scaling, 
                            targetX=b1000nib.shape[0], targetY=b1000nib.shape[1])
outputs[1], _ = autorescale(outputs[1], force_scaling=1/scaling, 
                            targetX=b1000nib.shape[0], targetY=b1000nib.shape[1])
outputs[2], _ = autorescale(outputs[2], force_scaling=1/scaling, 
                            targetX=b1000nib.shape[0], targetY=b1000nib.shape[1])

nib.save(nib.Nifti1Image(np.stack([outputs[0]*10,
                                  outputs[1]*10,
                                  outputs[2]],axis=3).astype(np.uint8), b1000nib.affine), 
            "/app/data/predictions.nii.gz")

t_rescaling_ended = datetime.now()

print("RGB computation")
rgb_data, rgb_nib = compute_color_nib(b1000nib, outputs[0], outputs[1], outputs[2], maxvizval=10)
nib.save(rgb_nib, "/app/data/heatmap.nii.gz")

t_heatmap_ended = datetime.now()

print("Creating synthesis")
max_per_line = 5
pos_thr = 5
width = 3
height = 3

volvox = np.prod(b1000nib.header.get_zooms())/(1000*scaling*scaling)
positive_slices = np.sum(outputs[1],axis=(0,1))
n_positive_slices = np.sum(positive_slices>pos_thr)
subset = rgb_data[:,:,positive_slices>pos_thr]
num_line = math.ceil(n_positive_slices/max_per_line) + 1
fig, axs = plt.subplots(num_line,max_per_line)
if num_line == 1:
    axs = [axs]
fig.set_size_inches(max_per_line*width, num_line*height)
plt.axis('off')
plt.tight_layout()
i = 0
k = 0
iline = 0
for line in axs:
    for ax in line:
        ax.set_xticks([])
        ax.set_yticks([])
        if iline == 0:
            if k == 0:
                ax.text(0.5,0.5,"For research",ha='center', va='bottom', fontsize=18, color="red")
                ax.text(0.5,0.5,"use only",ha='center', va='top', fontsize=18, color="red")
            if k == 1:
                ax.text(0.5,0.5,"Brain size",ha='center', va='bottom', fontsize=15, color="green")
                ax.text(0.5,0.5,str(round(brainmaskvox*volvox,2)) + "mL",
                        ha='center', va='top', fontsize=18)
            if k == 2:
                ax.text(0.5,0.5,"Blob size",ha='center', va='bottom', fontsize=15, color="blue")
                ax.text(0.5,0.5,str(round(blobmaskvox*volvox,2)) + "mL",
                        ha='center', va='top', fontsize=18)
            if k == 3:
                ax.text(0.5,0.5,"Flair Visibility Area",ha='center', va='bottom', fontsize=15, color="red")
                ax.text(0.5,0.5,str(round(vizmaskvox*volvox,2)) + "mL",
                        ha='center', va='top', fontsize=18)
            k += 1
        else:
            if i >= n_positive_slices: break
            ax.imshow(subset[:,::-1,i].transpose((1,0,2)))
            i += 1
    iline += 1
plt.savefig('/app/data/synthesis.png', bbox_inches='tight', dpi=300)

t_figure_ended = datetime.now()

with open("/app/data/synthesis.txt", "w") as synthfile:
    synthfile.write("BrainMaskVoxels="+str(brainmaskvox)+"\n")
    synthfile.write("StrokeBlobVoxels="+str(blobmaskvox)+"\n")
    synthfile.write("FlairVizVoxels="+str(vizmaskvox)+"\n")
    synthfile.write("VoxVolume="+str(volvox)+"\n")
    synthfile.write("BrainMaskVolume="+str(brainmaskvox*volvox)+"\n")
    synthfile.write("StrokeBlobVolume="+str(blobmaskvox*volvox)+"\n")
    synthfile.write("FlairVizArea="+str(vizmaskvox*volvox)+"\n")
    synthfile.write("Time-1-Load="+str((t_loaded_modules-t_first).total_seconds())+"\n")
    synthfile.write("Time-2-ScanningDir="+str((t_scanned_directory-t_loaded_modules).total_seconds())+"\n")
    synthfile.write("Time-3-Normalization="+str((t_data_normalized-t_scanned_directory).total_seconds())+"\n")
    synthfile.write("Time-4-Stacking="+str((t_data_stacked-t_data_normalized).total_seconds())+"\n")
    synthfile.write("Time-5-Inference="+str((t_inference_ended-t_data_stacked).total_seconds())+"\n")
    synthfile.write("Time-6-BackRescaling="+str((t_rescaling_ended-t_inference_ended).total_seconds())+"\n")
    synthfile.write("Time-7-Heatmap="+str((t_heatmap_ended-t_rescaling_ended).total_seconds())+"\n")
    synthfile.write("Time-8-Figure="+str((t_figure_ended-t_heatmap_ended).total_seconds())+"\n")
