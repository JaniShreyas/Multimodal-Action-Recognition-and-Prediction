from torchvision.transforms import Compose, Lambda, CenterCrop, Normalize
from ..transforms import FixedSizeClipSampler, TransformKey
import torch
import torch.nn as nn
import os
import pandas as pd
from ..config import DevConfig

def get_x3d_model():
    model_name = "x3d_s"
    model = torch.hub.load("facebookresearch/pytorchvideo", model_name, pretrained=True)

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    for param in model.blocks[-1].proj.parameters():
        param.requires_grad = True

    num_classes = len(
        pd.read_csv(os.path.join(DevConfig.ANNOTATIONS_DIR_LOCAL, "actions_label.csv"))
    )
    model.blocks[-1].proj = nn.Linear(
        in_features=model.blocks[-1].proj.in_features, out_features=num_classes
    )
    
    return model

def scale_pixels(x):
    return x / 255.0

def permute_tensor(x):
    return x.permute(1, 0, 2, 3)

def get_x3d_transform_compose():
    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]
    crop_size = 256
    data_transform = Compose(
        [
            FixedSizeClipSampler(num_frames=32),
            Lambda(scale_pixels),
            Normalize(mean, std),
            CenterCrop(crop_size),
            Lambda(permute_tensor),
        ]
    )
    data_transform = TransformKey("frames", data_transform)

    return data_transform