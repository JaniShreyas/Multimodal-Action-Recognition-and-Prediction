from .config import DevConfig
from .datasets import UCF50Dataset
from .models.X3D import get_x3d_model, get_x3d_transform_compose
import torch
from torch.utils.data import DataLoader
import os
import time

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
    model.load_state_dict(torch.load(os.path.join(DevConfig.MODELS_DIR_LOCAL, "model_initial.pth"), map_location=device))
    model.to(device)

    model.eval()

    # Evaluate on the test dataset
    total_samples = len(test_dataset)
    correct_predictions = 0
    inference_times = []

    with torch.no_grad():
        for batch in test_dataloader:
            inputs = batch["frames"].to(device)
            labels = batch["action_label"].to(device)

            start_time = time.perf_counter()

            outputs = model(inputs)
            predictions = outputs.argmax(dim = 1)

            end_time = time.perf_counter()
            inference_times.append(end_time - start_time)

            correct_predictions += (predictions == labels).sum().item()
    
    accuracy = (correct_predictions/total_samples) * 100
    avg_inference_time = sum(inference_times) / len(test_dataset)

    print("Test Accuracy: {:.2f}%".format(accuracy))
    print(f"Average Inference Time per sample: {avg_inference_time:.4f} seconds")