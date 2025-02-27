import os
from datetime import datetime

def create_log_dir(model_log_dir, experiment_name):
    # Create the experiment directory if it doesn't already exist.
    experiment_dir = os.path.join(model_log_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Generate a unique subdirectory name using the current timestamp.
    run_dir = os.path.join(experiment_dir, f'run_{datetime.now().strftime("%Y%m%d-%H%M%S")}')
    os.makedirs(run_dir, exist_ok=True)
    return run_dir
