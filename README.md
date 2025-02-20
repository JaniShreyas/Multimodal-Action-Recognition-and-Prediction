# Epic Kitchens 100 Multimodal Action Recognition & Prediction

This repository contains training code for the Epic Kitchens 100 dataset. In its current version, the code trains on RGB frames to predict verb classes. Future work will integrate optical flow frames, noun classes, and include fine-tuned weights for the X3D model. <br>
The src folder contains the actual python scripts for training the models, while the test_notebooks folder is for my personal testing throughout the process using notebooks.

> Note: Please change the data directory link in src/config.py to point to your Data directory instead. I have my data saved in a separate location and am thus using the absolute path for the Root directory.

## Setup Options

### Option 1: Install uv

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then run the following in the repo directory

```powershell
uv sync
```

### Option 2: Use requirements.txt

Create a python venv using 

```powershell
python -m venv .venv
```

Then activate using 

```powershell
.venv\Scripts\Activate.ps1
```

And then install the requirements using

```powershell
pip install -r requirements.txt
```

### Current Implementation

The project currently contains code for fine-tuning either X3D or SlowFast models using rgb_frames as input and verb_class as output
The plan is to also flow_frames for input and the list of noun_classes as output

Also, add the train_till_p105.csv annotation file in the annotations/ sub directory inside the main data directory. This is a small subset of the data on which I am currently testing models to see if they are working and will then eventually shift to a bigger subset.

Until then to start training using X3D, run

```powershell
uv run python -m src.train
```

or if not using uv, make sure the venv is active and run

```powershell
python -m src.train
```

> Note: If your device has low RAM, try to reduce num_workers in the dataloader in src/train.py before running
