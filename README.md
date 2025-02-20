# Epic Kitchens 100 Multimodal Action Recognition & Prediction

This repository contains training code for the Epic Kitchens 100 dataset. In its current version, the code trains on RGB frames to predict verb classes. Future work will integrate optical flow frames, noun classes, and include fine-tuned weights for the X3D model.

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

Until then to start training using X3D, run

```powershell
uv run python -m src.train
```

or 

```powershell
python -m src.train
```

> Note: If your device has low RAM, try to reduce num_workers in the dataloader in src/train.py before running
