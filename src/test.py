from .config import DevConfig
from .datasets import UCF50Dataset
from .models.X3D import get_x3d_model, get_x3d_transform_compose
import torch
from torch.utils.data import DataLoader
import os

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load test dataset
    test_dataset = UCF50Dataset(
        DevConfig.ROOT_DIR,
        DevConfig.TEST_ANNOTATIONS_FILE_LOCAL,
        transform = get_x3d_transform_compose()
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=2, shuffle=True, num_workers=4
    )

    model = get_x3d_model()
    model.load_state_dict(torch.load(os.path.join(DevConfig.MODELS_DIR_LOCAL, "model_temp.pth"), map_location=device))
    model.to(device)

    model.eval()

    # Evaluate on the test dataset
    total_samples = len(test_dataset)
    correct_predictions = 0

    with torch.no_grad():
        for batch in test_dataloader:
            inputs = batch["frames"].to(device)
            labels = batch["action_label"].to(device)

            outputs = model(inputs)
            
            predictions = outputs.argmax(dim = 1)
            correct_predictions += (predictions == labels).sum().item()
    
    accuracy = (correct_predictions/total_samples) * 100
    print("Test Accuracy: {:.2f}%".format(accuracy))