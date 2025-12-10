import subprocess
import sys

pipeline_steps = [
    ("Generating dummy data...", "python generate_dummy_data.py"),
    ("Preprocessing data...", "python preprocess_data.py"),
    ("Training model...", "python train_model.py"),
    ("Making predictions...", "python predict_demand.py"),
    ("Generating graphs...", "python output_graphs.py"),
]

for step_name, command in pipeline_steps:
    print(f"\n{'='*60}")
    print(f"{step_name}")
    print(f"{'='*60}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"Error in: {step_name}")
        sys.exit(1)

print(f"\n{'='*60}")
print("Pipeline complete")
print(f"{'='*60}")
