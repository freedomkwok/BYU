import os
import numpy as np
import pandas as pd
import random
import json
import glob
import math
import shutil
import time
import yaml
from pathlib import Path
from tqdm.notebook import tqdm  # Use tqdm.notebook for Jupyter/Kaggle environments
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from matplotlib.patches import Rectangle
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import cv2
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

import re
import optuna
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Define Kaggle paths
data_path = "../input/byu-locating-bacterial-flagellar-motors-2025/"
train_dir = os.path.join(data_path, "train")

# Define constants
TRUST = 4  # Number of slices above and below center slice (total 2*TRUST + 1 slices)
BOX_SIZE = 25  # Bounding box size for annotations (in pixels)
TRAIN_SPLIT = 0.98  # 98% for training, 2% for validation

# Local development paths
local_dev =  "/workspace/BYU/notebooks" if "WANDB_API_KEY" in os.environ else "C:/Users/Freedomkwok2022/ML_Learn/BYU/notebooks"
yolo_dataset_dir = os.path.join(local_dev, 'yolo_dataset')
yolo_weights_dir = os.path.join(local_dev, 'yolo_weights')
output_location = local_dev

# Create necessary directories
os.makedirs(yolo_weights_dir, exist_ok=True)

# Dataset directories
DATA_DIR = '../input/byu-locating-bacterial-flagellar-motors-2025'
TRAIN_CSV = os.path.join(DATA_DIR, 'labels.csv')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
OUTPUT_DIR = './'
MODEL_DIR = './models'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Set device: Use GPU if available; otherwise, fall back to CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Set random seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

import wandb

def plot_dfl_loss_curve(run_dir):
    """
    Plot the DFL loss curves for training and validation, marking the best model.
    
    Args:
        run_dir (str): Directory where the training results are stored.
    """
    results_csv = os.path.join(run_dir, 'results.csv')
    if not os.path.exists(results_csv):
        print(f"Results file not found at {results_csv}")
        return
    
    results_df = pd.read_csv(results_csv)
    train_dfl_col = [col for col in results_df.columns if 'train/dfl_loss' in col]
    val_dfl_col = [col for col in results_df.columns if 'val/dfl_loss' in col]
    
    if not train_dfl_col or not val_dfl_col:
        print("DFL loss columns not found in results CSV")
        print(f"Available columns: {results_df.columns.tolist()}")
        return
    
    train_dfl_col = train_dfl_col[0]
    val_dfl_col = val_dfl_col[0]
    
    best_epoch = results_df[val_dfl_col].idxmin()
    best_val_loss = results_df.loc[best_epoch, val_dfl_col]
    
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['epoch'], results_df[train_dfl_col], label='Train DFL Loss')
    plt.plot(results_df['epoch'], results_df[val_dfl_col], label='Validation DFL Loss')
    plt.axvline(x=results_df.loc[best_epoch, 'epoch'], color='r', linestyle='--', 
                label=f'Best Model (Epoch {int(results_df.loc[best_epoch, "epoch"])}, Val Loss: {best_val_loss:.4f})')
    plt.xlabel('Epoch')
    plt.ylabel('DFL Loss')
    plt.title('Training and Validation DFL Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plot_path = os.path.join(run_dir, 'dfl_loss_curve.png')
    plt.savefig(plot_path)
    plt.savefig(os.path.join(output_location, 'dfl_loss_curve.png'))
    
    print(f"Loss curve saved to {plot_path}")
    plt.close()
    
    return best_epoch, best_val_loss

def predict_on_samples(model, val_dir, num_samples=4, conf:float=0.25):
    """
    Run predictions on random validation samples and display results.
    
    Args:
        model: Trained YOLO model.
        num_samples (int): Number of random samples to test.
    """
    # val_dir os.path.join(yolo_dataset_dir, 'images', 'val')
    if not os.path.exists(val_dir):
        print(f"Validation directory not found at {val_dir}")
        val_dir = os.path.join(yolo_dataset_dir, 'images', 'train')
        print(f"Using train directory for predictions instead: {val_dir}")
        
    if not os.path.exists(val_dir):
        print("No images directory found for predictions")
        return
    
    val_images = os.listdir(val_dir)
    if len(val_images) == 0:
        print("No images found for prediction")
        return
    
    num_samples = min(num_samples, len(val_images))
    samples = random.sample(val_images, num_samples)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for i, img_file in enumerate(samples):
        if i >= len(axes):
            break
            
        img_path = os.path.join(val_dir, img_file)
        results = model.predict(img_path, conf=float(conf))[0]
        img = Image.open(img_path)
        axes[i].imshow(np.array(img), cmap='gray')
        
        # Draw ground truth box if available (extracted from filename)
        try:
            parts = img_file.split('_')
            y_part = [p for p in parts if p.startswith('y')]
            x_part = [p for p in parts if p.startswith('x')]
            if y_part and x_part:
                y_gt = int(y_part[0][1:])
                x_gt = int(x_part[0][1:].split('.')[0])
                rect_gt = Rectangle((x_gt - BOX_SIZE//2, y_gt - BOX_SIZE//2), BOX_SIZE, BOX_SIZE,
                                      linewidth=1, edgecolor='g', facecolor='none')
                axes[i].add_patch(rect_gt)
        except:
            pass
        
        if len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            for box, conf in zip(boxes, confs):
                x1, y1, x2, y2 = box
                rect_pred = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
                axes[i].add_patch(rect_pred)
                axes[i].text(x1, y1-5, f'{conf:.2f}', color='red')
        
        axes[i].set_title(f"Image: {img_file}\nGT (green) vs Pred (red)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_location, 'predictions.png'))
    plt.show()

def prepare_dataset(source):
    """
    Check if the dataset exists and create/fix a proper YAML file for training.
    
    Returns:
        str: Path to the YAML file to use for training.
    """
    train_images_dir = os.path.join(yolo_dataset_dir, 'images', source, 'train')
    val_images_dir = os.path.join(yolo_dataset_dir, 'images', source, 'val')
    train_labels_dir = os.path.join(yolo_dataset_dir, 'labels', source, 'train')
    val_labels_dir = os.path.join(yolo_dataset_dir, 'labels', source, 'val')
    
    # print(f"Directory status:")
    # print(f"- Train images exists: {os.path.exists(train_images_dir)}")
    # print(f"- Val images exists: {os.path.exists(val_images_dir)}")
    # print(f"- Train labels exists: {os.path.exists(train_labels_dir)}")
    # print(f"- Val labels exists: {os.path.exists(val_labels_dir)}")
    
    yaml_data = {
        'path': yolo_dataset_dir,
        'train': train_images_dir,
        'val': val_images_dir,
        'names': {0: 'motor'}
    }
    new_yaml_path = os.path.join(local_dev, 'dataset.yaml')
    with open(new_yaml_path, 'w') as f:
        yaml.dump(yaml_data, f)
    print(f"Created new YAML at {new_yaml_path}")
    print(yaml_data)
    
    return (new_yaml_path, train_images_dir, train_labels_dir, val_images_dir, val_labels_dir)

def get_latest_model_version(base_dir, prefix="motor_detector"):
    max_version = -1
    best_path = None

    for name in os.listdir(base_dir):
        if name.startswith(prefix):
            match = re.search(rf"{prefix}.*?optuna_trial_(\d+)", name)  # works with 'v1' or '1'
            if match:
                version_num = int(match.group(1))
                candidate_path = os.path.join(base_dir, name, "weights", "best.pt")
                if os.path.exists(candidate_path) and version_num > max_version:
                    max_version = version_num
                    best_path = candidate_path

    return max_version, best_path

def save_best_model_info(dataset_name, best_trial, version, weights_path):
    json_path = os.path.join(yolo_weights_dir, "best_models.json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            best_models = json.load(f)
    else:
        best_models = {}

    best_models[dataset_name] = {
        "version": version,
        "weights_path": weights_path,
        "val_loss": best_trial.value,
        "params": best_trial.params
    }

    with open(json_path, "w") as f:
        json.dump(best_models, f, indent=2)

def get_best_model_for_dataset(dataset_name):
    json_path = os.path.join(yolo_weights_dir, "best_models.json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f).get(dataset_name)
    return None

def clean_cuda_info():
    print(f"clean_cuda_info\n")
    if not torch.cuda.is_available():
        print("❌ CUDA is not available.")
        return

    torch.cuda.empty_cache()  # Clears unused memory from PyTorch cache
    torch.cuda.ipc_collect()  # Releases shared memory from inter-process comm
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()
    
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    
    num_devices = torch.cuda.device_count()
    print(f"✅ CUDA Available — {num_devices} device(s) detected.\n")
    
    for i in range(num_devices):
        print(f"🧠 Device {i}: {torch.cuda.get_device_name(i)}")
        print(f"  - Total Memory     : {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        print(f"  - Allocated Memory : {torch.cuda.memory_allocated(i) / 1e6:.2f} MB")
        print(f"  - Cached Memory    : {torch.cuda.memory_reserved(i) / 1e6:.2f} MB")
        print(f"  - Max Memory Alloc : {torch.cuda.max_memory_allocated(i) / 1e6:.2f} MB\n")

def run_optuna_tuning(dataset_name):
    study = optuna.create_study(
    study_name="yolo_hpo",
    storage="sqlite:///yolo_hpo.db",
    direction="minimize",
    load_if_exists=True,
)
    study.optimize(objective, n_trials=10)

    best_trial = study.best_trial
    best_version = f"motor_detector_{dataset_name}_optuna_trial_{best_trial.number}"
    best_weights_path = os.path.join(yolo_weights_dir, best_version, "weights", "best.pt")

    save_best_model_info(dataset_name, best_trial, best_version, best_weights_path)

    print("\n🎯 Best Trial:")
    print(f"  Version: {best_version}")
    print(f"  Value: {best_trial.value}")
    print(f"  Params: {best_trial.params}")
    print(f"  Weights: {best_weights_path}")
    
# --- Enhanced pretrained model selector ---
def select_pretrained_weights(dataset_name):
    best_model_info = get_best_model_for_dataset(dataset_name)
    latest_version, latest_best_path = get_latest_model_version(yolo_weights_dir)

    if best_model_info and os.path.exists(best_model_info["weights_path"]):
        print(f"✅ Using best saved model for '{dataset_name}': {best_model_info['weights_path']}")
        return best_model_info["weights_path"]
    # elif latest_best_path:
    #     print(f"🕓 No saved best model for '{dataset_name}'")
    #     print(f"latest available model: {latest_best_path}")
    #     return latest_best_path
    else:
        print("❌ No previous models found — using base weights: yolo11s.pt")
        return "yolo11s.pt"

def log_final_plots(run_dir: str):
    final_plots = [
        "F1_curve.png",
        "PR_curve.png",
        "P_curve.png",
        "R_curve.png",
        "dfl_loss_curve.png"
        "confusion_matrix.png"
    ]

    for plot_name in final_plots:
        plot_path = os.path.join(run_dir, plot_name)
        if os.path.exists(plot_path):
            wandb.log({plot_name: wandb.Image(plot_path)})
            
    df = pd.read_csv("results.csv")

    # Calculate F1
    df["metrics/F1(B)"] = 2 * df["metrics/precision(B)"] * df["metrics/recall(B)"] / (
        df["metrics/precision(B)"] + df["metrics/recall(B)"]
    )

    # Create a W&B table
    table = wandb.Table(columns=["epoch", "precision", "recall", "F1"])
    for _, row in df.iterrows():
        table.add_data(
            row["epoch"],
            row["metrics/precision(B)"],
            row["metrics/recall(B)"],
            row["metrics/F1(B)"]
        )

    # Log the full table once
    wandb.log({"F1_per_epoch": table})
           
            
def objective(trial):
    try:
        clean_cuda_info()
        
        version = f"motor_detector_{dataset_name}_optuna_trial_{trial.number}"
        run_dir = os.path.join(yolo_weights_dir, f"{version}")

        # run_dir = os.path.join(yolo_weights_dir, f"{version}", "runs")
        # os.makedirs(yolo_weights_dir, exist_ok=True)

        # Define custom callback
        def custom_epoch_end_callback(trainer):
            epoch = trainer.epoch
            metrics = trainer.metrics  # after validation step
            loss = trainer.loss_items  # training loss components: box, cls, dfl
            
            wandb.log({
                "epoch": epoch,
                "train/box_loss": loss[0],
                "train/cls_loss": loss[1],
                "train/dfl_loss": loss[2],
                "val/box_loss": metrics.get("val/box_loss", 0),
                "val/cls_loss": metrics.get("val/cls_loss", 0),
                "val/dfl_loss": metrics.get("val/dfl_loss", 0),
                "metrics/mAP50": metrics.get("metrics/mAP50", 0),
                "metrics/mAP50-95": metrics.get("metrics/mAP50-95", 0),
                "metrics/precision": metrics.get("metrics/precision", 0),
                "metrics/recall": metrics.get("metrics/recall", 0),
            })
            
            for k, v in trainer.metrics.items():
                print(k, v)
            
        trial_params = {
            "batch": trial.suggest_categorical("batch88", [88]), #600ada: 200 
            "imgsz": trial.suggest_categorical("imgsz", [512, 640]),
            # step 1
            "lr0": trial.suggest_float("lr0", 0.015, 0.02, log=True),
            "lrf": trial.suggest_float("lrf", 1e-2, 3e-1),
            "box": trial.suggest_float("box", 7, 8.5),   #7.7
            "cls": trial.suggest_float("cls", 0.01, 0.55), #0.55
            # "dfl": trial.suggest_float("dfl", 0.1, 1.3),
            # mosaic=trial.suggest_float("mosaic", 0.0, 0.4),
            # step 2
            # "scale": trial.suggest_float("scale", 0.0, 0.7),
            # "translate": trial.suggest_float("mosaic", 0.0, 0.4),
            # hsv_h=hsv_h,
            # hsv_s=hsv_s,
            # hsv_v=hsv_v,
            # "flipud": trial.suggest_float("flipud", 0.0, 1.0),
            # "fliplr": trial.suggest_float("fliplr", 0.0, 1.0),
            #bgr=trial.suggest_float("bgr", 0.0, 1.0),
            #mixup=trial.suggest_float("mixup", 0.0, 1.0),
        }

        wandb.init(
            project="BYU",
            name=f"trial_{trial.number}",
            config=trial_params,
            reinit=True
        )
        
        model = YOLO(pretrained_weights_path)
        model.add_callback("on_train_epoch_end", custom_epoch_end_callback)
        model.train(
            data=yaml_path,
            epochs=50,
            project=yolo_weights_dir,
            name=f"{version}",
            exist_ok=True,
            patience=6,
            verbose=False,
            amp=True,
            device=0,
            **trial_params
        )

        result = plot_dfl_loss_curve(run_dir)
        if result is None:
            wandb.finish()
            return float("inf")

        best_epoch, best_val_loss = result
        print(f"Trial {trial.number}: Best Val DFL Loss = {best_val_loss:.4f} at Epoch {best_epoch}")
        wandb.finish()
        return best_val_loss
         
    finally:
        wandb.finish()
        return float("inf")
        
def main():
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    global yaml_path, pretrained_weights_path, dataset_name
    print("Starting YOLO Optuna parameter tuning...")
    dataset_name = "shared"
 
    pretrained_weights_path = select_pretrained_weights(dataset_name)

    yaml_path, *_ = prepare_dataset("shared")

    run_optuna_tuning("shared")

if __name__ == "__main__":
    main()