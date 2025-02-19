import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .dataset import EpicKitchens100Dataset
from torchvision.transforms import Compose, Lambda
from torchvision.transforms import Normalize, CenterCrop
from .transforms import FixedSizeClipSampler, TransformKey, PackPathway

# Load the official SlowFast model from PyTorch Hub.
model = torch.hub.load("facebookresearch/pytorchvideo", "slowfast_r50", pretrained=True)
model.train()

mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 256
train_transform = Compose(
    [
        FixedSizeClipSampler(num_frames=32),
        Lambda(lambda x: x / 255.0),
        Normalize(mean, std),
        CenterCrop(crop_size),
        Lambda(lambda x: x.permute(1, 0, 2, 3)),
        PackPathway()
    ]
)
train_transform = TransformKey("frames", train_transform)
