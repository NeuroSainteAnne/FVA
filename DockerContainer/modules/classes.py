import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from torcheval.metrics import BinaryAccuracy
from dataclasses import dataclass, field, asdict
from typing import Tuple, List, Optional, Any, Dict
import json

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
    # Index for the stroke outline in the input data
    blob_index_ydat: int
    # Index for the flairviz outline in the input data
    viz_index_ydat: int
    # Index for the mask outline in the output data
    mask_index: int
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
        self.zcenter = int(np.where(np.array(self.zselector) == 0)[0][0])

    def to_dict(self):
        # Convert the dataclass to a dictionary
        data = asdict(self)
        data["zcenter"] = int(data["zcenter"]) # compatibility for np.int64
        
        # Handle non-serializable fields
        data['pat_ids'] = None
        data['train_dataloader'] = None  # DataLoader cannot be serialized directly
        data['valid_dataloader'] = None
        data['bn_dataloader'] = None
        data['simple_viz'] = None
        data['simple_noviz'] = None
        data['simple_mixviz'] = None
        data['run'] = None
        data['model'] = str(self.model) if self.model is not None else None
        data['device'] = str(self.device) if self.device is not None else None
        data['bce'] = str(self.bce) if self.bce is not None else None
        data['bce_weighted'] = str(self.bce_weighted) if self.bce_weighted is not None else None
        data['acc'] = str(self.acc) if self.acc is not None else None
        data['optimizer'] = str(self.optimizer) if self.optimizer is not None else None
        data['scheduler'] = str(self.scheduler) if self.scheduler is not None else None

        return data
        
    @classmethod
    def from_dict(cls, data: Dict):
        # Note: Other fields like model, dataloaders, optimizer, etc. cannot be reconstructed from strings
        # Here, they are set to None as a placeholder
        del data['cselector'] 
        del data['zcenter'] 
        data['pat_ids'] = None
        data['train_dataloader'] = None
        data['valid_dataloader'] = None
        data['bn_dataloader'] = None
        data['simple_viz'] = None
        data['simple_noviz'] = None
        data['simple_mixviz'] = None
        data['run'] = None
        data['model'] = None
        data['device'] = None
        data['bce'] = None
        data['bce_weighted'] = None
        data['acc'] = None
        data['optimizer'] = None
        data['scheduler'] = None

        return cls(**data)
        
    def to_json(self):
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str):
        return cls.from_dict(json.loads(json_str))