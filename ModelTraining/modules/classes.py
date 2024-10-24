import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from torcheval.metrics import BinaryAccuracy
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Any, Dict

# Definition of the TrainingObject class, which is used to encapsulate all training-related data and configurations
@dataclass
class TrainingObject:
    # Configuration dictionary containing training parameters
    config: Dict[str, Any]
    # Tuple of integers indicating which data channels to use as input (e.g., b1000, b0)
    xselector: Tuple[int, ...]
    # The number of input channels is derived from the length of xselector (initialized automatically)
    cselector: int = field(init=False)
    # Tuple of integers indicating which data channels to use as output (e.g., stroke, flairviz)
    yselector: Tuple[int, ...]
    # Index for the stroke outline in the output data
    blob_index: int
    # Index for the flairviz outline in the output data
    viz_index: int
    # List of z-slice indices used for input
    zselector: List[int]
    # Index of the center z-slice (initialized automatically)
    zcenter: int = field(init=False)
    # Current epoch for training
    epoch: int
    # DataFrame containing patient IDs and metadata
    pat_ids: pd.DataFrame 
    # model to be trained 
    model: Optional[Any] = field(default=None)
    # Device on which the model will be trained (e.g., CPU or GPU)
    device: Optional[Any] = field(default=None)
    # DataLoader for training data
    train_dataloader: Optional[DataLoader] = field(default=None)
    # DataLoader for validation data
    valid_dataloader: Optional[DataLoader] = field(default=None)
    # DataLoader for batch normalization
    bn_dataloader: Optional[DataLoader] = field(default=None)
    # DataLoader for visualization of simple examples during validation (e.g., fully visible areas)
    simple_viz: Optional[DataLoader] = field(default=None)
    # DataLoader for visualization of examples without visibility during validation
    simple_noviz: Optional[DataLoader]  = field(default=None)
    # DataLoader for mixed visualization examples during validation (e.g., partial visibility)
    simple_mixviz: Optional[DataLoader] = field(default=None)
    # Loss function without weighting, using SoftBCEWithLogitsLoss
    bce: Optional[smp.losses.SoftBCEWithLogitsLoss] = field(default=None)
    # Weighted loss function, using SoftBCEWithLogitsLoss
    bce_weighted: Optional[smp.losses.SoftBCEWithLogitsLoss] = field(default=None)
    # Metric for binary accuracy
    acc: Optional[BinaryAccuracy] = field(default=None)
    # Optimizer for the training process (e.g., SGD or Adam)
    optimizer: Optional[Any] = field(default=None)
    # Scheduler for adjusting learning rate during training
    scheduler: Optional[Any] = field(default=None)
    # Optional run object for tracking (e.g., with Weights & Biases)
    run: Optional[Any] = field(default=None)
    
    # Post-initialization method to set values derived from provided arguments
    def __post_init__(self):
        # Set the number of input channels to the length of xselector
        self.cselector = len(self.xselector)
        # Set the center z-slice index (i.e., the index where zselector is zero)
        self.zcenter = np.where(np.array(self.zselector) == 0)[0][0]