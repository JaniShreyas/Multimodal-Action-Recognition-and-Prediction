import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .datasets import UCF50Dataset
from torchvision.transforms import Compose, Lambda
from torchvision.transforms import Normalize, CenterCrop
from .transforms import FixedSizeClipSampler, TransformKey, PackPathway
from .config import DevConfig
import pandas as pd
import os
from torch.utils.tensorboard import SummaryWriter
from .utils.logs import create_log_dir


def scale_pixels(x):
    return x / 255.0


def permute_tensor(x):
    return x.permute(1, 0, 2, 3)


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()

    # Load the official SlowFast model from PyTorch Hub.
    model_name = "x3d_s"
    model = torch.hub.load("facebookresearch/pytorchvideo", model_name, pretrained=True)

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    for param in model.blocks[-1].proj.parameters():
        param.requires_grad = True

    log_dir = create_log_dir(DevConfig.LOGS_DIR_LOCAL, "X3D_only_freeze")

    # Create Tensorboard summary writer for logging
    writer = SummaryWriter(log_dir)

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

    train_dataset = UCF50Dataset(
        DevConfig.ROOT_DIR,
        DevConfig.TRAIN_ANNOTATIONS_FILE_LOCAL,
        transform=data_transform,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=2, shuffle=True, num_workers=4
    )

    val_dataset = UCF50Dataset(
        DevConfig.ROOT_DIR,
        DevConfig.VAL_ANNOTATIONS_FILE_LOCAL,
        transform=data_transform,
    )
    val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=True, num_workers=4)

    num_classes = len(
        pd.read_csv(os.path.join(DevConfig.ANNOTATIONS_DIR_LOCAL, "actions_label.csv"))
    )
    model.blocks[-1].proj = nn.Linear(
        in_features=model.blocks[-1].proj.in_features, out_features=num_classes
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        running_loss = 0.0
        batch_count = 0
        model.train()
        for batch in train_dataloader:
            if batch_count % 50 == 0:
                print(f"training on batch: {batch_count}")
            input = batch["frames"].to(device)
            batch_count += 1

            labels = batch["action_label"].to(device)

            # Forward pass
            outputs = model(input)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            global_step = epoch * len(train_dataloader) + batch_count

            writer.add_scalar("Loss/Train_Batch", loss.item(), global_step)

        avg_train_loss = running_loss / len(train_dataloader)
        writer.add_scalar("Loss/Train_Epoch", avg_train_loss, epoch)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                input = batch["frames"].to(device)
                labels = batch["action_label"].to(device)
                outputs = model(input)
                loss = criterion(outputs, labels)

                valid_loss += loss.item()

        avg_val_loss = valid_loss / len(val_dataloader)
        writer.add_scalar("Loss/Valid_Epoch", avg_val_loss, epoch)

        print(f"Validation Loss: {avg_val_loss:.4f}")

    # After finishing training
    torch.save(
        model.state_dict(),
        f"{DevConfig.MODELS_DIR_LOCAL}/model_temp.pth",
    )
    print(f"Model saved to {DevConfig.MODELS_DIR_LOCAL}")
