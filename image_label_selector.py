import os
import shutil
import glob
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import numpy as np

# === Configuration ===
images_dir = './yolo_dataset/images/shared'
labels_dir = './yolo_dataset/labels/shared'
move_target_dir = './selected'
move_bad_dir = './bad'

# === Create output folders if needed ===
os.makedirs(move_target_dir, exist_ok=True)
os.makedirs(move_bad_dir, exist_ok=True)

# === Gather all image paths ===
image_paths = sorted(glob.glob(os.path.join(images_dir, '*.jpg')))
current_index = 0
fig, ax = plt.subplots()


def draw_image_with_boxes(img_path):
    ax.clear()
    basename = os.path.basename(img_path)
    label_path = os.path.join(labels_dir, os.path.splitext(basename)[0] + '.txt')

    img = Image.open(img_path)
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
    global current_index

    if current_index >= len(image_paths):
        print("All images processed.")
        plt.close()
        return

    img_path = image_paths[current_index]
    basename = os.path.basename(img_path)
    label_path = os.path.join(labels_dir, os.path.splitext(basename)[0] + '.txt')

    if event.key == 'right':
        print(f"→ Moving {basename} and label to {move_target_dir}")
        shutil.move(img_path, os.path.join(move_target_dir, basename))
        if os.path.exists(label_path):
            shutil.move(label_path, os.path.join(move_target_dir, os.path.basename(label_path)))
    elif event.key == 'left':
        print(f"← Moving {basename} only to {move_bad_dir}")
        shutil.move(img_path, os.path.join(move_bad_dir, basename))
        # label is not moved

    current_index += 1

    if current_index < len(image_paths):
        draw_image_with_boxes(image_paths[current_index])
    else:
        print("All images processed.")
        plt.close()


# === Start viewer ===
if image_paths:
    draw_image_with_boxes(image_paths[current_index])
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()
else:
    print("No images found.")