import torch
import torch.nn as nn


class FixedSizeClipSampler(nn.Module):
    """
    A custom temporal sampler that:
      1) Pads short clips (N < 32) by repeating the last frame until 32.
      2) Uniformly subsamples long clips (N > 32) down to 32 frames.
    """

    def __init__(self, num_frames=32):
        super().__init__()
        self.num_frames = num_frames

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames (torch.Tensor): A 4D tensor of shape (T, C, H, W)
              - T: temporal dimension (# of frames)
              - C: # of channels (3 for RGB)
              - H, W: spatial dimensions

        Returns:
            A 4D tensor of shape (32, C, H, W) with fixed temporal length.
        """
        t = frames.shape[0]

        if t < self.num_frames:
            # --- Pad short clips ---
            pad_needed = self.num_frames - t
            last_frame = frames[-1:].clone()  # shape (1, C, H, W)
            # Repeat the last frame 'pad_needed' times and concatenate
            pad_frames = last_frame.repeat(pad_needed, 1, 1, 1)
            frames = torch.cat([frames, pad_frames], dim=0)

        elif t > self.num_frames:
            # --- Uniform subsampling for long clips ---
            # Create 32 indices evenly spaced from [0..t-1]
            indices = torch.linspace(0, t - 1, self.num_frames).long()
            frames = frames[indices]

        # If t == self.num_frames, we do nothing
        return frames
