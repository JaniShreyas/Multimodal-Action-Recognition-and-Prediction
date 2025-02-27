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
        
        video_name, action, action_label = row["video_name"], row["action"], row["action_label"]

        video_frames = io.read_video(os.path.join(self.root_dir, action, video_name), pts_unit='sec', output_format="TCHW")[0]
        

        sample = {
            "frames": video_frames,
            "action_label": action_label
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    @staticmethod
    def show_frames(frames, video_id, start_frame, narration):
        plt.figure(figsize=(8, 6))
        for i in range(len(frames)):
            img = frames[i].permute(1, 2, 0)
            plt.imshow(img)
            plt.title(f"Video: {video_id} | Frame {start_frame + i} | {narration}")
            plt.axis("off")
            plt.pause(0.1)  # display each frame for 0.1 seconds
            plt.clf()  # clear the figure for the next frame

        plt.close()