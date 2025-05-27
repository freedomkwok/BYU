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

images_dir_template = f'C:/Users/Freedomkwok2022/ML_Learn/BYU/notebooks/{source_folder}/{types}/{dateset}'
labels_dir_template = f'C:/Users/Freedomkwok2022/ML_Learn/BYU/notebooks/{source_folder}/{types}/{dateset}'

selected_images_dir = images_dir_template.replace(types, "images").replace(source_folder, "selected")
selected_labels_dir = images_dir_template.replace(types, "labels").replace(source_folder, "selected")
bad_images_dir = images_dir_template.replace(types, "images").replace(source_folder, "bad")
bad_labels_dir = images_dir_template.replace(types, "labels").replace(source_folder, "bad")

os.makedirs(selected_images_dir, exist_ok=True)
os.makedirs(selected_labels_dir, exist_ok=True)
os.makedirs(bad_images_dir, exist_ok=True)
os.makedirs(bad_labels_dir, exist_ok=True)

# === Gather all image paths ===
images_dir = os.path.join(images_dir_template.replace(types, "images").replace(source_folder, "yolo_dataset"))
labels_dir = os.path.join(labels_dir_template.replace(types, "labels").replace(source_folder, "yolo_dataset"))

print("images_dir: ", images_dir)
image_paths = sorted(glob.glob(os.path.join(images_dir, '**', '*.jpg'), recursive=True))
current_index = 0
checkpoint_file = '_image_viewer_checkpoint.json'

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
history_stack = deque(maxlen=50)
last_click = None
box_size_ratio = 0.07  # 7% of width/height

def save_checkpoint():
    if 0 <= current_index < len(image_paths):
        with open(checkpoint_file, 'w') as f:
            json.dump({
                "index": current_index,
                "image_path": image_paths[current_index]
            }, f)

def draw_image_with_boxes(img_path):
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
                x = (x_c - w / 2) * img_width
                y = (y_c - h / 2) * img_height
                width = w * img_width
                height = h * img_height

                rect = Rectangle((x, y), width, height, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(x, y - 5, f"Class {int(class_id)}", color='red', fontsize=8)
    else:
        ax.text(10, 10, "No annotations found", color='red', fontsize=12)

    fig.canvas.draw()

def on_key(event):
    global current_index, last_click

    if event.key == 'right':
        if current_index >= len(image_paths):
            print("All images processed.")
            plt.close()
            return

        img_path = image_paths[current_index]
        basename = os.path.basename(img_path)

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
            new_filename = f"{prefix}_z{z:04d}_x{int(x):04d}_y{int(y):04d}_w{int(img_width):04d}_h{int(img_height):04d}_r{box_size_tag}.jpg"

            new_img_path = os.path.join(selected_images_dir, new_filename)
            new_lbl_path = os.path.join(selected_labels_dir, os.path.splitext(new_filename)[0] + '.txt')

            shutil.copy(img_path, new_img_path)

            with open(new_lbl_path, 'w') as f:
                f.write(f"0 {norm_x:.16f} {norm_y:.16f} {norm_w:.3f} {norm_h:.3f}\n")

            history_stack.append((new_filename, 'selected', True))
        else:
            shutil.copy(img_path, os.path.join(selected_images_dir, basename))
            history_stack.append((basename, 'selected', False))

        last_click = None
        current_index += 1

    elif event.key == 'up':
        if current_index >= len(image_paths):
            print("All images processed.")
            plt.close()
            return

        img_path = image_paths[current_index]
        basename = os.path.basename(img_path)

        print(f"⬆ Moving {basename} only to {bad_images_dir}")
        shutil.move(img_path, os.path.join(bad_images_dir, basename))
        history_stack.append((basename, 'bad', False))
        last_click = None
        current_index += 1

    elif event.key == 'left':
        if not history_stack:
            print("Nothing to undo.")
            return

        current_index -= 1
        last_basename, folder_type, label_moved = history_stack.pop()
        print(f"⏪ Undoing {last_basename} from {folder_type}")

        if folder_type == 'selected':
            try:
                image_path = os.path.join(selected_images_dir, last_basename)
                if os.path.exists(image_path):
                    os.remove(image_path)
                if label_moved:
                    label_file = os.path.splitext(last_basename)[0] + '.txt'
                    label_path = os.path.join(selected_labels_dir, label_file)
                    if os.path.exists(label_path):
                        os.remove(label_path)
            except Exception as e:
                print("Error undoing selected:", e)
        elif folder_type == 'bad':
            try:
                os.remove(os.path.join(bad_images_dir, last_basename))
            except Exception as e:
                print("Error undoing bad:", e)

        last_click = None

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
