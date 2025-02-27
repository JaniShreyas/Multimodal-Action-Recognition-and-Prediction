class DevConfig:
    ROOT_DIR = r"C:\Users\Jani\.cache\kagglehub\datasets\vineethakkinapalli\ucf50-action-recognition-dataset\versions\1\UCF50"
    ANNOTATIONS_DIR_LOCAL = "annotations"
    FULL_ANNOTATIONS_FILE_LOCAL = "annotations/annotations.csv"  # Locations of annotations directory in root directory
    TRAIN_ANNOTATIONS_FILE_LOCAL = "annotations/train_annotations.csv"
    VAL_ANNOTATIONS_FILE_LOCAL = "annotations/val_annotations.csv"
    TEST_ANNOTATIONS_FILE_LOCAL = "annotations/test_annotations.csv"
    MODELS_DIR_LOCAL = "outputs/models"
    RANDOM_STATE = 42