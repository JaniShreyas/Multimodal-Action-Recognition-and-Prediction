import os
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.io as io

# Currently only for the rgb_frames along with narration
class EpicKitchens100Dataset(Dataset):
    def __init__(self, root_dir, annotations_csv, transform = None):
        self.root_dir = root_dir
        self.annotations_df = pd.read_csv(os.path.join(self.root_dir, annotations_csv))
        self.transform = transform
    
    def __len__(self):
        return len(self.annotations_df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.annotations_df.iloc[idx]
        video_id, start_frame, stop_frame, narration, verb_class = row[["video_id", "start_frame", "stop_frame", "narration", "verb_class"]]
        participant_id = video_id.split("_")[0]
        frames_dir = os.path.join(self.root_dir, participant_id, "rgb_frames")
        frames = self.get_frames_for_segment(frames_dir, video_id, start_frame, stop_frame)
        sample = {"video_id": video_id, "frames": frames, "narration": narration, "verb_class": verb_class}

        if self.transform:
            sample = self.transform(sample)
        
        return sample

    def get_frame_path(self, frames_dir, video_id, frame_num):
        filename = f"frame_{frame_num:010d}.jpg"
        return os.path.join(frames_dir, video_id, filename)

    def get_frames_for_segment(self, frames_dir, video_id, start_frame, stop_frame):
        frames = []
        for i in range(start_frame, stop_frame + 1):
            frame_path = (self.get_frame_path(frames_dir, video_id, i))
            if os.path.exists(frame_path):
                frame = io.read_image(frame_path)
                frames.append(frame)
            else:
                print(f"Missing frame: {i}")
        return torch.stack(frames, dim = 0)

    @staticmethod
    def show_frames(frames, video_id, start_frame, narration):
        plt.figure(figsize=(8, 6))
        for i in range(len(frames)):
            img = frames[i].permute(1,2,0)
            plt.imshow(img)
            plt.title(f"Video: {video_id} | Frame {start_frame + i} | {narration}")
            plt.axis("off")
            plt.pause(0.1)  # display each frame for 0.1 seconds
            plt.clf()       # clear the figure for the next frame

        plt.close()
