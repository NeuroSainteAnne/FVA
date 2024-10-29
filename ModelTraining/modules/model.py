import torch
import segmentation_models_pytorch_3d as smp3d
import segmentation_models_pytorch as smp
from modules.classes import TrainingObject
import os

def create_FVV_model(status, device=None):
    ### MODEL LOADING
    if status.config["preload_model"] is not None:
        previous_status_path = status.config["preload_model"].split("-epoch")[0]+".config.json"
        if os.path.exists(previous_status_path): # check compatibility mode
            with open(previous_status_path, "r") as pv: 
                previous_status = TrainingObject.from_json(pv.read())
        else: # compatibility mode for older model without config
            model = torch.load(status.config["preload_model"], weights_only=False).module
            if device is None:
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.decoder.to(device)
            model.encoder.to(device)
            model.segmentation_head.to(device)
            return model, device
            
    if status.config["input_2.5D"] == "smp3d":
        ### 3D model
        if status.config["model_name"] == "Unet": smpmod = smp3d.Unet
        elif status.config["model_name"] == "Unet++": smpmod = smp3d.UnetPlusPlus
        elif status.config["model_name"] == "MANet": smpmod = smp3d.MAnet
        elif status.config["model_name"] == "LinkNet": smpmod = smp3d.Linknet
        elif status.config["model_name"] == "FPN": smpmod = smp3d.FPN
        elif status.config["model_name"] == "PSPNet": smpmod = smp3d.PSPNet
        elif status.config["model_name"] == "PAN": smpmod = smp3d.PAN
        elif status.config["model_name"] == "DeepLabV3": smpmod = smp3d.DeepLabV3
        elif status.config["model_name"] == "DeepLabV3+": smpmod = smp3d.DeepLabV3Plus
        model = smpmod(
            encoder_name=status.config["encoder_name"],        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=status.config["encoder_weights"],  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=status.cselector,                      # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=len(status.yselector)+1,                   # model output channels (number of classes in your dataset)
            strides=((1, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2)),
        )
        ### Segmentation head for 2d output
        if status.config["model_name"] == "Unet": smpmod2d = smp.Unet
        elif status.config["model_name"] == "Unet++": smpmod2d = smp.UnetPlusPlus
        elif status.config["model_name"] == "MANet": smpmod2d = smp.MAnet
        elif status.config["model_name"] == "LinkNet": smpmod2d = smp.Linknet
        elif status.config["model_name"] == "FPN": smpmod2d = smp.FPN
        elif status.config["model_name"] == "PSPNet": smpmod2d = smp.PSPNet
        elif status.config["model_name"] == "PAN": smpmod2d = smp.PAN
        elif status.config["model_name"] == "DeepLabV3": smpmod2d = smp.DeepLabV3
        elif status.config["model_name"] == "DeepLabV3+": smpmod2d = smp.DeepLabV3Plus
        model2D = smpmod2d(
            encoder_name=status.config["encoder_name"],        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=status.config["encoder_weights"],     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=status.cselector,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=len(status.yselector)+1,                      # model output channels (number of classes in your dataset)
        )
        model.decoder = model2D.decoder
        model.segmentation_head = model2D.segmentation_head

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if status.config["preload_model"]:
        model.load_state_dict(torch.load(status.config["preload_model"], weights_only=True))
            
    ## load model on GPU
    if status.config["multigpu"]:
        model = torch.nn.DataParallel(model)
    model.to(device)
    return model, device
    