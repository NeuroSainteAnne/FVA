import torch
import segmentation_models_pytorch_3d as smp3d
import segmentation_models_pytorch as smp
from modules.classes import TrainingObject
import os

class QATOverhead(torch.nn.Module):
    def __init__(self, original_model):
        super().__init__()
        # QuantStub converts tensors from floating point to quantized
        self.quant = torch.ao.quantization.QuantStub()
        self.encoder = original_model.encoder
        self.decoder = original_model.decoder
        self.seghead0 = original_model.segmentation_head[0]
        self.seghead1 = original_model.segmentation_head[1]
        # DeQuantStub converts tensors from quantized to floating point
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        features = self.encoder(x)
        features = [i[:,:,j] for i,j in zip(features,[3, 3, 2, 2, 1, 1])] # 3D to 2D
        x = self.decoder(*features)
        x = self.seghead0(x)
        x = self.seghead1(x)
        x = self.dequant(x)
        return x

def qat_integrate(model, device):
    model_fp32 = QATOverhead(model)
    model_fp32.eval()
    model_fp32.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')
    model_fp32_prepared = torch.ao.quantization.prepare_qat(model_fp32.train())
    return model_fp32_prepared
    
def create_FVV_model(status, device=None):
    ### MODEL LOADING
    preloaded_model = False
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
        preloaded_model = True
            
    if status.config["input_2.5D"] == "smp3d" and not preloaded_model:
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
    
    if status.config["preload_model"] and not preloaded_model:
        model.load_state_dict(torch.load(status.config["preload_model"], weights_only=True))

            
    if "qat_finetune" in status.config and status.config["qat_finetune"]:
        model = qat_integrate(model, device)
            
    ## load model on GPU
    if status.config["multigpu"]:
        model = torch.nn.DataParallel(model)
    model.to(device)
    return model, device
    