import os
from PIL import Image
import torch
import clip
import numpy as np
from sklearn.cluster import KMeans

# === CONFIG ===
image_folder = "C:/path/to/your/image_folder"  # ← Change this
output_file = "image_clusters.txt"
n_clusters = 5  # You can tune this

# === Setup ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# === Load images ===
image_paths = [
    os.path.join(image_folder, f)
    for f in os.listdir(image_folder)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
]

print(f"Found {len(image_paths)} images.")

# === Extract CLIP features ===
features = []
valid_paths = []

for path in image_paths:
    try:
        img = preprocess(Image.open(path).convert("L")).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(img)
        features.append(feat.cpu().numpy())
        valid_paths.append(path)
    except Exception as e:
        print(f"❌ Skipping {path}: {e}")

features_np = np.vstack(features)

# === Cluster features ===
kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(features_np)
labels = kmeans.labels_

# === Save result to file ===
with open(output_file, 'w') as f:
    for path, label in zip(valid_paths, labels):
        f.write(f"{os.path.basename(path)} {label}\n")

print(f"\n✅ Done. Total clusters: {n_clusters}")
print(f"📄 Output saved to: {output_file}")
