from .config import DevConfig
from .datasets import UCF50Dataset
from .models.X3D import get_x3d_model, get_x3d_transform_compose
import torch
import os

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    
    # Load test dataset
    test_dataset = UCF50Dataset(
        DevConfig.ROOT_DIR,
        DevConfig.TEST_ANNOTATIONS_FILE_LOCAL,
        transform = get_x3d_transform_compose()
    )

    model = get_x3d_model()

    model.load_state_dict(torch.load(os.path.join(DevConfig.MODELS_DIR_LOCAL, "model_temp.pth")))
    print(model)