import os
import shutil
import glob
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import numpy as np
from collections import deque
import re

# === Configuration ===
types = "[types]"
source_folder = "[source]"
dateset = "shared_007"

local_dev = "/Users/freedomkwokmacbookpro/Github/Kaggle/BYU/notebooks" if os.path.exists("/Applications") else "C:/Users/Freedomkwok2022/ML_Learn/BYU/notebooks/"
images_dir_template = f'{local_dev}/{source_folder}/{types}/{dateset}'
labels_dir_template = f'{local_dev}/{source_folder}/{types}/{dateset}'

cleaned_images_dir = images_dir_template.replace(types, "images").replace(source_folder, "cleaned") + "/train"
cleaned_labels_dir = images_dir_template.replace(types, "labels").replace(source_folder, "cleaned") + "/train"

os.makedirs(cleaned_images_dir, exist_ok=True)
os.makedirs(cleaned_labels_dir, exist_ok=True)

from_folder = "selected"
# === Gather all image paths ===
images_dir = images_dir_template.replace(types, "images").replace(source_folder, from_folder)
labels_dir = images_dir_template.replace(types, "labels").replace(source_folder, from_folder)

print("images_dir: ", images_dir)
image_paths = sorted(glob.glob(os.path.join(images_dir, '**', '*.jpg'), recursive=True))
total = len(image_paths)
current_index = 0
checkpoint_file = '_image_cleaner_checkpoint.json'
last_box_info =None

# === Load checkpoint if exists ===
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, 'r') as f:
        checkpoint = json.load(f)
        last_path = checkpoint.get("image_path")
        if last_path in image_paths:
            current_index = image_paths.index(last_path)
            print(f"⏪ Resuming from: {last_path} (index {current_index})")
        else:
            print("⚠️ Checkpoint image not found. Starting from beginning.")

fig, ax = plt.subplots()
history_stack = deque(maxlen=450)
last_click = None
defaut_box_size_ratio = 0.09  # 7% of width/height
box_size_ratio = defaut_box_size_ratio

def save_checkpoint():
    if 0 <= current_index < len(image_paths):
        with open(checkpoint_file, 'w') as f:
            json.dump({
                "index": current_index,
                "image_path": image_paths[current_index]
            }, f)
    
def draw_image_with_boxes(img_path):
    global last_box_info
    ax.clear()
    basename = os.path.basename(img_path)
    parent = os.path.basename(os.path.dirname(img_path))
    basename = os.path.join(parent, basename)
    label_path = os.path.join(labels_dir, os.path.splitext(basename)[0] + '.txt')

    img = Image.open(img_path).convert('L')
    img_width, img_height = img.size

    img_array = np.array(img)
    p2, p98 = np.percentile(img_array, 2), np.percentile(img_array, 98)
    normalized = np.clip(img_array, p2, p98)
    normalized = 255 * (normalized - p2) / (p98 - p2)
    img_normalized = Image.fromarray(np.uint8(normalized)).convert('RGB')

    ax.imshow(img_normalized)
    ax.set_title(basename)
    ax.axis('off')

    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                class_id, x_c, y_c, w, h = map(float, line.strip().split())
                x_center = x_c * img_width if last_click is None else last_click[0]
                y_center = y_c * img_height if last_click is None else last_click[1]
                
                ratio_changed = (box_size_ratio != defaut_box_size_ratio)
                
                width = w * img_width if not ratio_changed else box_size_ratio* img_width
                height = h * img_height if not ratio_changed else box_size_ratio* img_height
                x = x_center - width / 2
                y = y_center - height / 2

                print("\n[image info]")
                print("center:" , "old:",[x_center , y_center], [x_c, y_c], "image_size" ,[img_width, img_height])
                print("box:", "old:", [width, height], [w,h], "cur", [box_size_ratio, box_size_ratio])
                
                rect = Rectangle((x, y), width, height, linewidth=2, edgecolor=('r' if not ratio_changed else 'lime'), facecolor='none')
                ax.add_patch(rect)
                ax.text(x, y - 5, f"{box_size_ratio}", color=('red' if not ratio_changed else 'lime'), fontsize=8)
    else:
        ax.text(10, 10, "No annotations found", color='red', fontsize=12)

    fig.canvas.draw()

def reset():
    global last_click, box_size_ratio
    last_click = None
    # box_size_ratio = defaut_box_size_ratio
    
def get_prefix(name):
    match = re.match(r'^(.*)_z\d+', name)
    return match.group(1) if match else name
    
def on_key(event):
    global current_index, last_click, box_size_ratio
    
    print(f"{current_index}/{total}/n")
    
    if event.key == 'pageup':
        box_size_ratio = min(1.0, box_size_ratio + 0.01)
        print(f"🔍 Enlarged box size: {box_size_ratio:.3f}")

    elif event.key == 'pagedown':
        box_size_ratio = max(0.01, box_size_ratio - 0.01)
        print(f"🔎 Shrunk box size: {box_size_ratio:.3f}")

    elif event.key == 'home':
        box_size_ratio = defaut_box_size_ratio
        print("↩️ Box size reset to default {defaut_box_size_ratio}")

    elif event.key == 'up':
        current_index += 1
        img_path = image_paths[current_index]
        basename = os.path.basename(img_path)
        history_stack.append((basename, 'selected', False))
            
    elif event.key == 'right':
        if current_index >= len(image_paths):
            print("All images processed.")
            plt.close()
            return

        img_path = image_paths[current_index]
        basename = os.path.basename(img_path)
        print(last_click)
        if last_click:
            x, y, img_width, img_height = last_click
            norm_x = x / img_width
            norm_y = y / img_height
            norm_w = box_size_ratio
            norm_h = box_size_ratio

            prefix_match = re.match(r'^(.*)_z(\d+)', basename)
            if prefix_match:
                prefix = prefix_match.group(1)
                z = int(prefix_match.group(2))
            else:
                prefix = os.path.splitext(basename)[0]
                z = current_index

            box_size_tag = f"{box_size_ratio:.3f}".replace(".", "")[-3:]
            new_filename = f"{prefix}_z{z:04d}_y{int(y):04d}_x{int(x):04d}_w{int(img_width):04d}_h{int(img_height):04d}_r{box_size_tag}.jpg"

            new_img_path = os.path.join(cleaned_images_dir, new_filename)
            new_lbl_path = os.path.join(cleaned_labels_dir, os.path.splitext(new_filename)[0] + '.txt')

            shutil.copy(img_path, new_img_path)
            print(f"🔁{x, y} {norm_x, norm_y} Manual-copied with box: {new_filename} {x, y}")
            
            with open(new_lbl_path, 'w') as f:
                f.write(f"0 {norm_x:.16f} {norm_y:.16f} {norm_w:.3f} {norm_h:.3f}\n")
            print("newImage to:", new_img_path )
            print("newLabel to:", new_lbl_path )
            
            history_stack.append((new_filename, 'selected', True))
        else:
            new_cleaned_image = os.path.join(cleaned_images_dir, basename)
            shutil.copy(img_path, new_cleaned_image)
            print(f"✅ Copied old to: {new_cleaned_image}")
            
            old_label_file = img_path.replace("images", "labels").replace("jpg", "txt")
            
            if os.path.exists(old_label_file):
                new_label_path = os.path.join(cleaned_labels_dir, os.path.basename(old_label_file))
                print("copy_old_label", new_label_path)
                shutil.copy(old_label_file, new_label_path)
                
            print("label from: ", old_label_file)    
            history_stack.append((basename, 'selected', False))
        
        current_index += 1 #so we moved and this is the current new image
 
        reset()
        
    elif event.key == 'left':
        if not history_stack:
            print("Nothing to undo.")
            return

        current_index -= 1
        last_basename, folder_type, label_moved = history_stack.pop()
        print(f"⏪ Undoing {last_basename} from {folder_type}")

        if folder_type == 'selected':
            try:
                reverted_image_path = os.path.join(cleaned_images_dir, last_basename)
                if os.path.exists(reverted_image_path):
                    print("revert image file: ", reverted_image_path)
                    os.remove(reverted_image_path)

                label_file = os.path.splitext(last_basename)[0] + '.txt'
                reverted_label_file = os.path.join(cleaned_labels_dir, label_file)
                if os.path.exists(reverted_label_file):
                    os.remove(reverted_label_file)
                    print("revert label file: ", reverted_label_file)
            except Exception as e:
                print("Error undoing selected:", e)

        reset()

    save_checkpoint()
    if 0 <= current_index < len(image_paths):
        draw_image_with_boxes(image_paths[current_index])
    else:
        print("All images processed.")
        plt.close()

def on_click(event):
    global last_click, current_index

    if event.inaxes != ax:
        return

    x, y = event.xdata, event.ydata
    if x is None or y is None:
        return

    img = ax.images[0].get_array()
    img_height, img_width = img.shape[:2]
    box_w = img_width * box_size_ratio
    box_h = img_height * box_size_ratio
    x0 = x - box_w / 2
    y0 = y - box_h / 2

    if event.button == 1:  # Left click → draw box
        last_click = (x, y, img_width, img_height)
        [p.remove() for p in ax.patches]
        rect = Rectangle((x0, y0), box_w, box_h, linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        fig.canvas.draw()
        print(f"🟩 Box drawn at ({int(x)}, {int(y)})")

# === Start viewer ===
if image_paths:
    draw_image_with_boxes(image_paths[current_index])
    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()
else:
    print("No images found.")
