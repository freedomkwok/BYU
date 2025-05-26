import os
import shutil
import zipfile
from glob import glob

base = "/workspace/BYU/notebooks"
source_base = os.path.join(base, "yolo_weights")
target_base = os.path.join(base, "results")

# Ensure output directory exists
os.makedirs(target_base, exist_ok=True)

def zip_folder(folder_path, zip_path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, start=folder_path)
                zipf.write(abs_path, rel_path)
                
# Find all trial folders like *_trial*/
trial_dirs = [d for d in glob(os.path.join(source_base, "*_trial*")) if os.path.isdir(d)]

for trial_path in trial_dirs:
    trial_name = os.path.basename(trial_path)
    dest_dir = os.path.join(target_base, trial_name)
    os.makedirs(dest_dir, exist_ok=True)

    # Collect all relevant files
    for ext in ("*.png", "*.jpg", "*.csv"):
        for file_path in glob(os.path.join(trial_path, ext)):
            try:
                shutil.copy(file_path, dest_dir)
                print(f"Copied: {file_path} → {dest_dir}")
            except Exception as e:
                print(f"Failed to copy {file_path}: {e}")
                
zip_folder(target_base, target_base + '.zip')

print("\n✅ All trials copied and zipped into ./results/")

