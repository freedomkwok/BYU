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
from functools import partial
import argparse
import yaml

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
yolo_models_dir = os.path.join(local_dev, 'models')

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
import gc
last_time = time.time()
gpu_name = [torch.cuda.get_device_name(i).replace("NVIDIA GeForce ", "") for i in range(torch.cuda.device_count())][0]

def plot_dfl_loss_curve(run_dir):
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
    
    print(f"Loss curve saved to {plot_path}")
    plt.close()
    
    return best_epoch, best_val_loss

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
    
    gc.collect()
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

def run_optuna_tuning(dataset_name, args):
    args.storage = args.storage or "yolo_hpo"
    storage_name = f'sqlite:///{args.storage}.db'
    
    study_name = None
    if args.study:
        study_name = args.study
    elif args.saved_model is not None:
        study_name = "manual_test"
    else:
        study_name = "yolo_hpo"
        
    args.study = study_name
     
    resume = args.resume if args.resume is not None else False
    n_trials = 1 if args.saved_model is not None or resume else 120
    
    print(f"🎯Loading Study: {study_name} storage:{storage_name} \n")
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="minimize",
        load_if_exists=True,
    )
    study.optimize(partial(objective, args = args), n_trials=n_trials)

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
def select_pretrained_weights(dataset_name, custom_model = None):
    if custom_model:
        yolo_models_dir = os.path.join(local_dev, 'models')
        return os.path.join(yolo_models_dir, f"{custom_model}.yaml")
    
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
        print("❌ No previous models found — using base weights: yolo11m.pt")
        return "yolo11m.pt"
           
def compute_f1_score(precision, recall):
    if (precision + recall) == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def read_yaml(saved_model, resume):
    fixed_args = {}
    with open(os.path.join(yolo_weights_dir, saved_model, "args.yaml"), "r") as f:
        fixed_args = yaml.safe_load(f)
    
    model_str= None
    weight_path = os.path.join(yolo_weights_dir, saved_model, "weights/best.pt")
    if not resume: # note this doesnt read the model jsut the yaml
        model_str = fixed_args.get("model", "")
    else:
        model_str = weight_path
    
    return fixed_args, model_str
        
def objective(trial, args):
    try:
        clean_cuda_info()
        dataset_name = args.dataset 
        study = args.study
        saved_model = args.saved_model
        resume = args.resume
        custom_model = args.custom_model
        report_epoch  = args.report_epoch
        report_every  = args.report_every
        
        trial.set_user_attr("frozen_layer", args.frozen_layer)
        frozen_layer_index = args.frozen_layer - 1
        
        frozen_epoch = trial.suggest_int("frozen_epoch", args.frozen_epoch, args.frozen_epoch + 20)
        unfreeze_every = trial.set_user_attr("unfreeze_every", args.unfreeze_every)
        pre_optimizer_name = args.pre_opt
        optimizer_name = args.opt
        
        args_dict = vars(args)
        print("🧨args:", args_dict)
        
        trial_params = {
            "batch": trial.suggest_categorical("batch16", [16]), #600ada: 200 88
            "imgsz": trial.suggest_categorical("imgsz640", [640]),
            "patience": trial.suggest_int("patience", 8, 18),
            # step 1
            # "lr0": trial.suggest_float("lr0", 0.0087, 0.0095, log=True),
            # "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-4),  # Regularization
            "lrf": trial.suggest_float("lrf", 0.04, 0.09),
            "box": trial.suggest_float("box", 9.45, 9.8),   #7.7
            "cls": trial.suggest_float("cls", 0.05, 0.2), #0.55
            "dfl": trial.suggest_float("dfl", 0.05, 1.8),
            # "momentum": trial.suggest_float("momentum", 0.4, 0.98),   # For SGD or Adam
            "mosaic": trial.suggest_float("mosaic", 0.11575, 0.13),
            "warmup_epochs": trial.suggest_int("warmup_epochs", 5, 15),
            # step 2
            "translate": trial.suggest_float("translate", 0.115, 0.125),
            "hsv_h": trial.suggest_float("hsv_h", 0.0, 0.015),
            "hsv_s": trial.suggest_float("hsv_s", 0.0, 0.2),
            "flipud": trial.suggest_float("flipud", 0.0, 0.4),
            "fliplr": trial.suggest_float("fliplr", 0.08, 0.4),
            "bgr": trial.suggest_float("bgr", 0.8, 1.0),
            "mixup": trial.suggest_float("mixup", 0.2, 0.7),
            "cutmix": trial.suggest_float("cutmix", 0.05, 0.35),
            "epochs": trial.suggest_int("epochs", 100, 160),
            "degrees": trial.suggest_float("degrees", 0.0, 25.0),
            "close_mosaic": trial.suggest_int("close_mosaic", frozen_epoch + 10, frozen_epoch + 100),
        }
        
        def suggest_optimizer_params(trial):
            trial.set_user_attr("optimizer", pre_optimizer_name)
            if pre_optimizer_name == "AdamW":
                weight_decay = trial.suggest_float("weight_decay", 1e-5, 0.05)
                optimizer_params = {
                    "optimizer": pre_optimizer_name,
                    "lr0": trial.suggest_float("lr0", 1e-5, 1e-3, log=True),
                    "weight_decay": weight_decay
                }
            else:  # SGD fallback or other optimizer
                momentum = trial.suggest_float("momentum", 0.4, 0.98)
                optimizer_params = {
                    "optimizer": pre_optimizer_name,
                    "lr0": trial.suggest_float("lr0", 0.0087, 0.0095, log=True),
                    "momentum": momentum
            }
                
            return optimizer_params

        _009_6000ada_trial_params = trial_params
        _yaml, _model = read_yaml(saved_model, resume) if saved_model is not None else (None, None) ##could be None
        trial_params = _yaml if saved_model is not None else  {**_009_6000ada_trial_params, **suggest_optimizer_params(trial)}
        _pretrained_weights_path = _model if saved_model is not None else pretrained_weights_path
        print("_pretrained_weights_path", _pretrained_weights_path)
        
        os.environ["WANDB_DISABLE_ARTIFACTS"] = "true"
        os.environ["WANDB_DISABLE_CODE"] = "true"  
        os.environ["WANDB_CONSOLE"] = "off"  

        batch_num = trial_params["batch"]
        image_size = trial_params["imgsz"]
        optimizer = trial_params["optimizer"]
        epochs = trial_params["epochs"]
        model = YOLO(_pretrained_weights_path)
        model_base = custom_model or model.yaml.get('yaml_file').replace(".yaml", "") or _pretrained_weights_path.replace(".pt")
        
        version = None
        if resume:
            version = _yaml.get("name", "resumed_run")
        else:
            version = f"{dataset_name}_{model_base}_{batch_num}_{epochs}_{trial.number}"
            version_dir = os.path.join(yolo_weights_dir, f"{version}")
            os.makedirs(version_dir, exist_ok=True)
            # trial_params.pop('save_dir', None)
            # trial_params.pop('name', None)
            
        addtional_configs = {"study":study, "_model_base": model_base, "_device": gpu_name, "_dataset_name": dataset_name}
        print("model:", model_base, _pretrained_weights_path)
        
        default_args = {
            "data":yaml_path,
            "project":yolo_weights_dir,
            "name":f"{version}",
            "exist_ok":True,
    	    # "single_cls"=True,
            "verbose":False,
            "amp":True,
            # "multi_scale"=True, #memory costly
            "cos_lr":True, #memory costly
            "device":0,
            "resume":resume,
            **trial_params,
        }
        
        wandb.init(
            project="BYU",
            name=f"{trial.number}",
            tags=[study, dataset_name, f'imgsz_{image_size}', f'batch_{batch_num}',f'{optimizer}', gpu_name, model_base],
            config=addtional_configs | default_args | args_dict,
            reinit=True
        )
                # Define custom callback
        def custom_epoch_end_callback(trainer):
            now = time.time()
        
            if not hasattr(trainer, 'last_time'):
                trainer.last_time = now

            epoch_time = now - trainer.last_time
            trainer.last_time = now
            steps = len(trainer.train_loader)
            batch_size = trainer.args.batch
            samples = steps * batch_size
            
            epoch = trainer.epoch
            metrics = trainer.metrics  # after validation step
            loss = trainer.loss_items  # training loss components: box, cls, dfl
            precision = metrics.get("metrics/precision(B)", 0.01)
            recall = metrics.get("metrics/recall(B)", 0.99)
            mAP95 = metrics.get("metrics/mAP50-95(B)", 0)
            f1 = compute_f1_score(precision, recall)
            # trial.report(mAP95, step=epoch)
            
            lr_dict = {}
            for i, pg in enumerate(trainer.optimizer.param_groups):
                lr = pg.get("lr", None)
                initial_lr = pg.get("initial_lr", None)
                
                if lr is not None:
                    lr_dict[f"lr/{i}"] = lr
                if initial_lr is not None:
                    lr_dict[f"initial_lr/{i}"] = initial_lr
                    
            wandb.log({
                "epoch": epoch,
                "train/box_loss": loss[0],
                "train/cls_loss": loss[1],
                "train/dfl_loss": loss[2],
                "val/box_loss": metrics.get("val/box_loss", 0),
                "val/cls_loss": metrics.get("val/cls_loss", 0),
                "val/dfl_loss": metrics.get("val/dfl_loss", 0),
                "metrics/mAP50": metrics.get("metrics/mAP50(B)", 0),
                "metrics/mAP50-95": mAP95,
                "metrics/precision": metrics.get("metrics/precision(B)", 0),
                "metrics/recall": metrics.get("metrics/recall(B)", 0),
                "metrics/f1": f1,
                "samples_per_second": samples / epoch_time if epoch > 1 else 0,
                "samples_trained": samples,
                "time_per_epoch": epoch_time,
                **lr_dict
            })

            if epoch >= report_epoch and epoch % report_every == 0:
                try:
                    model.plots()
                    best_epoch, best_val_loss = plot_dfl_loss_curve(version_dir)
                    print(f"📈 [Epoch {epoch}] Best Val DFL Loss so far: {best_val_loss:.4f} at epoch {best_epoch}")
                except Exception as e:
                    print(f"⚠️ Failed to plot DFL loss curve at epoch {epoch}: {e}")
            
            gc.collect()
            
        def rebuild_optimizer(trainer):
            if optimizer_name == 'AdamW':
                trainer.optimizer = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, trainer.model.parameters()),
                    lr=trainer.args.lr0,
                    betas=(0.9, 0.999),
                    weight_decay=getattr(trainer.args, 'weight_decay', 5e-4),
                )
                print("🔁 Rebuilt optimizer with AdamW")
            else:
                trainer.optimizer = torch.optim.SGD(
                    filter(lambda p: p.requires_grad, trainer.model.parameters()),
                    lr=trainer.args.lr0,
                    momentum=getattr(trainer.args, 'momentum', 0.9),
                    weight_decay=getattr(trainer.args, 'weight_decay', 5e-4),
                )
                print("🔁 Rebuilt optimizer with SGD")

            for pg in trainer.optimizer.param_groups:
                if "initial_lr" not in pg:
                    pg["initial_lr"] = pg["lr"]
            
        def freeze_all_backbone(model, max_backbone_idx=4):
            for i in range(max_backbone_idx + 1):
                for param in model.model[i].parameters():
                    param.requires_grad = False
            print(f"❄️ Frozen model.model[0] to model.model[{max_backbone_idx}]")
        
        def unfreeze_one_layer_back(model, idx):
            for param in model.model[idx].parameters():
                param.requires_grad = True
            print(f"🔓 Unfroze model.model[{idx}]")
        
        def custom_epoch_start_callback(trainer):
            epoch = trainer.epoch
           
            # 🧨 Skip all freezing logic if frozen_layer is -1
            if frozen_layer_index < 0:
                return

            # Initial freeze (run once)
            if not hasattr(trainer, "_unfrozen_idx"):
                trainer._unfrozen_idx = frozen_layer_index
                freeze_all_backbone(trainer.model, max_backbone_idx=trainer._unfrozen_idx)
                rebuild_optimizer(trainer)

            # Gradually unfreeze one block every N epochs
            if epoch >= frozen_epoch and (epoch - frozen_epoch) % unfreeze_every == 0:
                if trainer._unfrozen_idx >= 0:
                    unfreeze_one_layer_back(trainer.model, trainer._unfrozen_idx)
                    trainer._unfrozen_idx -= 1
                    if epoch == frozen_epoch:
                        rebuild_optimizer(trainer)
     
        model.add_callback("on_train_epoch_start", custom_epoch_start_callback)        
        model.add_callback("on_train_epoch_end", custom_epoch_end_callback)    
            
        model.train(**default_args)
        
        default_args.update(dict(resume=True, multi_scale=True, scale=0.5))
        model.train(**default_args)
        
        result = plot_dfl_loss_curve(version_dir)
        
        wandb.finish()
        
        if result is None:
            return float("inf")

        best_epoch, best_val_loss = result
        print(f"Trial {trial.number}: Best Val DFL Loss = {best_val_loss:.4f} at Epoch {best_epoch}")
   
        return best_val_loss
         
    finally:
        print(f"finally")

def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Optuna Tuning")
    parser.add_argument("--study", type=str, help="(Optional) study name", default="shared_010_scaled")
    parser.add_argument("--storage", type=str, help="(Optional) storage name", default="shared_010_scaled")
    parser.add_argument("--dataset", type=str, help="(Optional) Dataset name", default="shared_010_scaled")
    parser.add_argument("--epochs", type=str, help="(Optional) epochs")
    parser.add_argument("--saved_model", type=str, help="(Optional) saved_model")
    parser.add_argument("--resume", type=bool, default = False, help="resume")
    parser.add_argument("--custom_model", type=str, help="(Optional) custom_model", default="b5")
    parser.add_argument("--pre_opt", type=str, help="(Optional) custom_model", default="SGD")
    parser.add_argument("--opt", type=str, help="(Optional) custom_model", default="AdamW")
    parser.add_argument("--f_layer", dest="frozen_layer", type=int, default=5, help="Number of layer to freeze the backbone (default: 0)")
    parser.add_argument("--f_epoch", dest="frozen_epoch", type=int, default=20, help="Number of epochs to freeze the backbone (default: 0)")
    parser.add_argument("--uf_every", dest="unfreeze_every", type=int, default=5, help="Number of epoch to unfreeze a layer the backbone (default: 0)")
    parser.add_argument("--report_epoch", dest="report_epoch", type=int, default=50, help="Number of epoch to unfreeze a layer the backbone (default: 0)")
    parser.add_argument("--report_every", dest="report_every", type=int, default=10, help="Number of epoch to unfreeze a layer the backbone (default: 0)")
    return parser.parse_args()

def setup_wandb():
    if "WANDB_API_KEY" not in os.environ:
        os.environ["WANDB_MODE"] = "disabled"
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    
def main():
    args = parse_args()
    setup_wandb()
    
    global yaml_path, pretrained_weights_path
    print("Starting YOLO Optuna parameter tuning...")
    dataset_name = args.dataset or "shared_007"

    pretrained_weights_path = select_pretrained_weights(dataset_name, args.custom_model)  ## load weight

    yaml_path, *_ = prepare_dataset(dataset_name) ## load file

    run_optuna_tuning(dataset_name, args) ##save weight

if __name__ == "__main__":
    main()
