import os
import glob
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import numpy as np

types = "[types]"
source_folder = "[source]"
dateset = "shared_007"

# === Config ===
images_dir_template = f'C:/Users/Freedomkwok2022/ML_Learn/BYU/notebooks/{source_folder}/{types}/{dateset}'
labels_dir_template = f'C:/Users/Freedomkwok2022/ML_Learn/BYU/notebooks/{source_folder}/{types}/{dateset}'

selected_images_dir = images_dir_template.replace(types, "images").replace(source_folder, "selected")
selected_labels_dir = images_dir_template.replace(types, "labels").replace(source_folder, "selected")

image_paths = sorted(glob.glob(os.path.join(selected_images_dir, '*.jpg')))
current_index = 0

fig, ax = plt.subplots()


def draw_image_with_label(img_path):
    ax.clear()
    basename = os.path.basename(img_path)
    label_path = os.path.join(selected_labels_dir, os.path.splitext(basename)[0] + '.txt')

    img = Image.open(img_path).convert('L')
    img_width, img_height = img.size
    img_array = np.array(img)

    ax.imshow(img_array, cmap='gray')
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
                rect = Rectangle((x, y), width, height, linewidth=2, edgecolor='lime', facecolor='none')
                ax.add_patch(rect)
                ax.text(x, y - 5, f"Class {int(class_id)}", color='lime', fontsize=8)
    else:
        ax.text(10, 10, "No label file found", color='red')

    fig.canvas.draw()


def on_key(event):
    global current_index
    print(image_paths[current_index])
    if event.key == 'right':
        current_index += 1
        if current_index < len(image_paths):
            draw_image_with_label(image_paths[current_index])
        else:
            print("Reached end of selected images.")
            plt.close()
            
    elif event.key == 'left':
        current_index -= 1
        if current_index < len(image_paths):
            draw_image_with_label(image_paths[current_index])
        else:
            print("Reached end of selected images.")
            plt.close()

# === Start ===
if image_paths:
    draw_image_with_label(image_paths[current_index])
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()
else:
    print("No selected images found.")