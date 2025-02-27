import os
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.io as io


# Currently only for the rgb_frames along with narration
class UCF50Dataset(Dataset):
    def __init__(self, root_dir, annotations_csv, transform=None):
        self.root_dir = root_dir
        self.annotations_df = pd.read_csv(annotations_csv)
        self.transform = transform

    def __len__(self):
        return len(self.annotations_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.annotations_df.iloc[idx]

        video_name, action, action_label = (
            row["video_name"],
            row["action"],
            row["action_label"],
        )

        video_frames = io.read_video(
            os.path.join(self.root_dir, action, video_name),
            pts_unit="sec",
            output_format="TCHW",
        )[0]

        sample = {"frames": video_frames, "action_label": action_label}

        if self.transform:
            sample = self.transform(sample)

        return sample
