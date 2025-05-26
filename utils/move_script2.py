import os
import shutil

local_dev =  "/workspace/BYU/notebooks" if "WANDB_API_KEY" in os.environ else "C:/Users/Freedomkwok2022/ML_Learn/BYU/notebooks"
yolo_dataset_dir2 = os.path.join(local_dev, "yolo_dataset2")
yolo_dataset_dir = os.path.join(local_dev, "yolo_dataset")
image_source = os.path.join(yolo_dataset_dir2, "images")
label_source = os.path.join(yolo_dataset_dir2, "labels")

image_dst = os.path.join(yolo_dataset_dir, "images")
label_dst = os.path.join(yolo_dataset_dir, "labels")

os.makedirs( image_dst, exist_ok=True)
os.makedirs( label_dst, exist_ok=True)

# Create shared/{train,val}/ folders if not exist
for split in ["train", "val"]:
    os.makedirs(os.path.join(image_dst, "shared_007_100", split), exist_ok=True)
    os.makedirs(os.path.join(label_dst, "shared_007_100", split), exist_ok=True)

import numpy as np
def copy_dataset_to_shared(root_path, shared_path, dataset_names, split_types):
    for dataset in dataset_names:
        for split in split_types:
            src_dir = os.path.join(root_path, dataset, "val")
            dst_dir = os.path.join(shared_path, "val")
            if not os.path.exists(src_dir):
                print(f"❌ Source not found: {src_dir}")
                continue
            
            forAllFile = os.listdir(src_dir)
            all_indices = np.arange(len(forAllFile))
            selected_indices = np.random.choice(all_indices, size=int(len(forAllFile) * 1), replace=False)
            selected_files = [forAllFile[i] for i in selected_indices]
             
            for filename in selected_files[:2000]: 
                src_file = os.path.join(src_dir, filename)
                new_filename = f"{dataset}_{filename}"
                dst_file = os.path.join(dst_dir, new_filename)
                shutil.copy2(src_file, dst_file)
                print(f"✅ Copied: {src_file} -> {dst_file}")

# Define datasets and split types
datasets = ["cryoet_007"]
splits = ["train"]

# datasets = ["cryoet_007"]
# splits = ["val"]

# Copy for images
print("📁 Copying images...")
copy_dataset_to_shared(image_source, os.path.join(image_dst, "shared_007_100"), datasets, splits)

# Copy for labels
print("📁 Copying labels...")
copy_dataset_to_shared(label_source, os.path.join(label_dst, "shared_007_100"), datasets, splits)

print("✅ All files copied to shared/train and shared/val folders.")


