import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .dataset import EpicKitchens100Dataset
from torchvision.transforms import Compose, Lambda
from torchvision.transforms import Normalize, CenterCrop
from .transforms import FixedSizeClipSampler, TransformKey, PackPathway
from .config import DevConfig

# Load the official SlowFast model from PyTorch Hub.
model_name = "x3d_s"
model = torch.hub.load("facebookresearch/pytorchvideo", model_name, pretrained=True)
model


def scale_pixels(x):
    return x / 255.0


def permute_tensor(x):
    return x.permute(1, 0, 2, 3)


mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 256
train_transform = Compose(
    [
        FixedSizeClipSampler(num_frames=32),
        Lambda(scale_pixels),
        Normalize(mean, std),
        CenterCrop(crop_size),
        Lambda(permute_tensor),
    ]
)
train_transform = TransformKey("frames", train_transform)


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    train_dataset = EpicKitchens100Dataset(
        DevConfig.ROOT_DIR,
        DevConfig.ANNOTATIONS_DIR_RELATIVE,
        transform=train_transform,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=2, shuffle=True, num_workers=4
    )

    # ------------------------------------------------------
    # Currently hardcoded and temporary. To be changed later
    # ------------------------------------------------------
    num_classes = 88
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
        for batch in train_dataloader:
            if batch_count % 50 == 0:
                print(f"training on batch: {batch_count}")
            input = batch["frames"].to(device)
            batch_count += 1

            labels = batch["verb_class"].to(device)

            # Forward pass
            outputs = model(input)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_dataloader):.4f}"
        )

    # After finishing training
    torch.save(
        model.state_dict(),
        f"{DevConfig.MODELS_DIR_LOCAL}/model_temp_with_rgb_frames_and_verb_classes.pth",
    )
    print(f"Model saved to {DevConfig.MODELS_DIR_LOCAL}")
