import torch 
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision 
from torchvision.transforms import v2
from torchvision.transforms import functional as TF
import numpy as np
import random

### Definition of Dataset generator

class FVVDataset(Dataset):
    def __init__(self, xd, maskd, metad, yd=None, status=None, input25D=False, xselector=None, zselector=None):
        # Initialize the dataset with input data, mask, metadata, and output labels
        self.xdata = xd
        self.mask = maskd
        self.meta = metad
        self.ydata = yd
        
        # Set configuration parameters depending on whether status is provided (for training) or not (for inference)
        if status is not None:
            self.config = status.config
            self.transform = self.config["augmentation"]  # Whether to apply data augmentation
            self.xselector = status.xselector  # Selectors for input data channels
            self.yselector = status.yselector  # Selectors for output data channels
            self.zselector = status.zselector  # Selectors for z-slices
        else:
            self.input25D = input25D  # Whether to use 2.5D input
            self.xselector = xselector  # Selectors for input data channels
            self.zselector = zselector  # Selectors for z-slices
            self.transform = False  # No augmentation for inference
    
    # Function to apply transformations (data augmentation) to the input data, mask, and output
    def apply_transform(self, x, mask, y):
        # Apply random brightness and contrast adjustments
        minx = x.view(x.shape[0], -1).min(-1)[0]
        for i in range(x.shape[0]):
            bright = (torch.rand(1) - 0.5) * self.config["aug_brightness"]
            contr = 1 + (torch.rand(1) - 0.5) * self.config["aug_contrast"]
            x[i] = ((x[i] - minx[i]) * contr) + bright
        
        # Apply random vertical flip with 50% probability
        if random.random() > 0.5:
            x = TF.vflip(x)
            mask = TF.vflip(mask)
            y = TF.vflip(y)
        
        # Apply random affine transformations (rotation, translation, scaling, shearing)
        affineP = v2.RandomAffine.get_params(
            (-self.config["aug_angle"], self.config["aug_angle"]),
            (self.config["aug_translate"], self.config["aug_translate"]),
            (1 / self.config["aug_scale"], self.config["aug_scale"]),
            (-self.config["aug_shear"], self.config["aug_shear"]),
            img_size=x.shape[1:3]
        )
        x = TF.affine(x, affineP[0], affineP[1], affineP[2], affineP[3], 
                      interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
        mask = TF.affine(mask, affineP[0], affineP[1], affineP[2], affineP[3])
        y = TF.affine(y, affineP[0], affineP[1], affineP[2], affineP[3])
        
        # Add minimum value back to each channel
        for i in range(x.shape[0]):
            x[i] = x[i] + minx[i]
        
        return x, mask, y
    
    # Function to retrieve an item (data point) from the dataset
    def __getitem__(self, index):
        # Extract input data based on selector
        x = self.xdata[index, self.xselector]
        
        # Handle 2.5D input by stacking neighboring slices
        if (hasattr(self, 'config') and self.config["input_2.5D"]) or (not hasattr(self, 'config') and self.input25D):
            stacked = []
            paddingval = torch.tensor(np.ones_like(x) * np.min(x))[:, None]
            for slindx in self.zselector:
                # Handle boundary conditions by padding
                if index + slindx < 0 or index + slindx >= self.meta.shape[0] - 1:
                    stacked.append(paddingval)
                elif self.meta[index, 0] != self.meta[index + slindx, 0]:
                    stacked.append(paddingval)
                else:
                    stacked.append(torch.tensor(self.xdata[index + slindx, self.xselector])[:, None])
            
            # Apply random z-axis flipping if enabled for training
            if hasattr(self, 'config') and self.config.get("aug_z", False):
                if random.random() > 0.5:
                    stacked.reverse()
            x = torch.cat(stacked, 1)
        else:
            x = torch.tensor(x)
        
        # Extract mask and metadata
        mask = self.mask[index]
        meta = self.meta[index]
        
        # Extract output labels if available
        if self.ydata is not None:
            y = self.ydata[index, self.yselector] if hasattr(self, 'yselector') else self.ydata[index]
        else:
            y = None
        
        # Apply data augmentation if enabled (for training)
        if self.transform and y is not None:
            x, mask, y = self.apply_transform(x.float(), torch.tensor(mask), torch.tensor(y))
        
        # Return input, mask, metadata, and output labels (if available)
        if y is not None:
            return x.float(), mask, meta, y
        else:
            return x.float(), mask, meta
    
    # Function to get the length of the dataset
    def __len__(self):
        return len(self.xdata)

