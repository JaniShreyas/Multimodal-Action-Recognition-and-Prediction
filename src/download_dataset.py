import kagglehub

# Download latest version
path = kagglehub.dataset_download("vineethakkinapalli/ucf50-action-recognition-dataset")

print("Path to dataset files:", path)