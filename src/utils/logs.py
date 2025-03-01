import os
from datetime import datetime


def create_model_and_log_dir(model_dir, model_log_dir, experiment_name):
    # Create the experiment directory if it doesn't already exist.
    experiment_log_dir = os.path.join(model_log_dir, experiment_name)
    os.makedirs(experiment_log_dir, exist_ok=True)

    experiment_dir = os.path.join(model_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    run_name = f"run_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    # Generate a unique subdirectory name using the current timestamp.
    run_log_dir = os.path.join(
        experiment_log_dir, run_name
    )
    os.makedirs(run_log_dir, exist_ok=True)

    run_model_name = os.path.join(experiment_name, run_name + ".pth")

    return run_model_name, run_log_dir