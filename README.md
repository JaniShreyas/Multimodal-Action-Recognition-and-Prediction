# Multimodal Action Recognition and Prediction

This project aims to define a pipeline for training and fine-tuning a wide variety of Action recognition, prediction, and anticipation models on just as large a variety of datasets, including UCF50, and EpicKitchens100 (EK100).
Currently, the implementation includes a pipeline for training X3D on UCF50, as well as an implementation of EK100. The training and testing script are currently hard-set to UCF50 but can use EK100 as well by fiddling a little with the code.

The present model was trained on 70% data as training, 15% as validation, and 15% as testing, and gives ```98.60%``` accuracy with an average inference time of ```0.0135 seconds``` per sample with each batch containing 2 samples (batch_size = 2).<br>
The respective annotation csv files are present in the repo.

This repository contains training code for the UCF50 dataset. The code contains 4 main directories at the top level: src, experiment_notebooks, outputs, and annotations. <br>
* The ```src/``` directory contains the actual code for training and testing models (currently for X3D but will implement SlowFast and a few others in the future). <br>
* ```experiment_notebooks/``` contains rough initial prototyping/testing of the features that were then polished and modularized inside ```src/``` <br>
* ```outputs/``` contains the training logs, trained models, and predictions <br>
* ```annotations/``` contains the annotations files for reading the dataset stored elsewhere in the user's pc or locally (can be set in ```src/config.py```) <br>

If at any point in the setup, training, or testing process, you encounter any error or problem, you can contact me. (though there shouldn't be any problems other than the older EK100 notebooks)

> Note: The older notebooks under ```experiment_notebooks/old_ek100_notebooks/``` aren't configured correctly since the dataset has changed to UCF100 for now, so you can look at the outputs but some fiddling, again, will be needed to get them to work again.
> Although, there is not much to see in them currently.

## Setup Instructions

### Install uv
This project uses uv for package management: https://github.com/astral-sh/uv

Follow the instructions in the above repo link or run the following command in powershell to install uv

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then run the following in the repo directory to setup the packages

```powershell
uv sync
```

### Data Setup

#### Option 1: Included download script

The repo contains a download script called ```download_dataset.py``` under ```src/utils/```. The script will download the dataset in ```<user>/.cache/kagglehub/datasets/``` (where ```<user>``` is your ```C:\Users\user``` directory in windows). <br>
You can run the script using the following command in the root directory of the project:

```powershell
uv run python -m src.utils.download_dataset
```

This will download the dataset and output the path. You then have to copy the path, example: ```r"C:\Users\Jani\.cache\kagglehub\datasets\vineethakkinapalli\ucf50-action-recognition-dataset\versions\1\UCF50"```, and update the ROOT_DIR field in src.config's DevConfig class
Example:
```python
class DevConfig:
    ROOT_DIR = r"C:\Users\Jani\.cache\kagglehub\datasets\vineethakkinapalli\ucf50-action-recognition-dataset\versions\1\UCF50"
    ANNOTATIONS_DIR_LOCAL = "annotations"
    FULL_ANNOTATIONS_FILE_LOCAL = "annotations/annotations.csv"
    TRAIN_ANNOTATIONS_FILE_LOCAL = "annotations/train_annotations.csv"
    VAL_ANNOTATIONS_FILE_LOCAL = "annotations/val_annotations.csv"
    TEST_ANNOTATIONS_FILE_LOCAL = "annotations/test_annotations.csv"
    MODELS_DIR_LOCAL = "outputs/models"
    LOGS_DIR_LOCAL = "outputs/logs"
    RANDOM_STATE = 42
```

#### Option 2: Manual download
If you want to, you can download the dataset from here directly: https://www.kaggle.com/datasets/vineethakkinapalli/ucf50-action-recognition-dataset by clicking the ```download``` button and then clicking download dataset as zip. I have not used this personally but it should work (If it doesn't please use the download script)

After this, follow the same steps as above and update the config file with the path of the dataset. Local paths should work if your dataset is in the root directory under a folder like ```data/``` or something, but, again, I have not tested it, so try using the absolute path in that case as well.

## Current Implementation

The code currently contains off the shelf implementation for fine-tuning X3D using UCF50. The metrics evaluated currently are accuracy and inference time per sample which are ```98.60%``` and ```0.0135 seconds``` respectively (I will be adding different averaged F1 scores in a while as well).

The model present under outputs/models named model_temp.pth was trained on 70% of the data as training, 15% validation, and 15% testing.

The model weights were all frozen except for the last linear classification layer, and it was updated to have 50 classes instead of the 400 of kinetics it was trained with.

The training logs for the model are under ```outputs/logs/X3D_only_freeze/run_20250228-021754/``` and you can visualize them using tensorboard by running 
```powershell
uv run tensorboard --logdir="[absolute path of the aforementioned folder in your machine]"
```
in a separate terminal window and then going on the ```scalars``` tab for better visuals.

### How to train

For now, you can manually make whatever changes you want regarding the data and model in ```src/train.py``` and then run
```powershell
uv run python -m src.train
```
in the root directory of the project to start training the model. Each batch's training loss, as well as each epoch's average training and validation losses will be logged so copy the absolute path of the log directory under ```outputs/logs/[Experiment_name]/[time_stamp]/``` (like the one mentioned previously) and run the tensorboard command in a different terminal


### How to test

Similar to the training code, you can change the model you want to test in the code (for now; I will be adding command line args to choose the model to test in a while), and then run the following:
```powershell
uv run python -m src.test
```

And this will output the metrics.

### Other features

There is also a script to create annotation files and split them into train, valid, test csv files if you want to change the split size. It is ```src/utils/create_annotations_files.py``` and can be run with a similar command as the ```download_dataset``` one.
