
napari
热图降低8倍
Gaussian Heatmap

✅ 反卷积（Transposed Convolution）
  用于“放大”特征图
  例如：[8 × 8] → 反卷积 → [16 × 16]
  在 图像生成模型（如GAN）中很常见

✅ RNN（Recurrent Neural Network）
  用于建模 时间/序列依赖性
  例如：输入一段句子，预测下一个词。



🧠 How does it work?
1. Label creation:
    You generate a 3D Gaussian heatmap from (cx, cy, cz) — this becomes your supervised ground truth Y_true, shaped like [D, H, W].

H(x, y) = exp(-((x - μx)² + (y - μy)²) / (2σ²))
2. Model output:
    Your model outputs a prediction Y_pred, also shaped like [D, H, W], which tries to mimic the heatmap — ideally peaking at the same spot.

3. Loss function:
oss Type
What it does
MSELoss (L2)
Penalizes pixel-wise squared difference:
loss = (Y_pred - Y_true)^2

1 You generate a target heatmap from (cx, cy, cz) with a Gaussian
2 Your model outputs a predicted heatmap
3 You use MSE or similar loss to measure:
    how close is prediction to ground truth at every voxel?
4 The model learns to put its own peak near the real center
5 At inference, you can use argmax() to find the predicted center
6 You can compare pred_center vs true_center for localization error

SmoothBCE  Binary Cross Entropy Loss (BCE) is used for binary classification
Standard BCE formula:
\text{BCE}(p, y) = -[y \cdot \log(p) + (1 - y) \cdot \log(1 - p)]
Where:
	•	p is the predicted probability (from sigmoid)
	•	y is the ground truth (0 or 1)

def smooth_bce(eps=0.1):
    return 1.0 - 0.5 * eps, 0.5 * eps  # positive_label, negative_label
pos, neg = smooth_bce(0.1)  # pos=0.95, neg=0.05

🧊 What is Smooth BCE (or Label Smoothing for BCE)?
📌 Problem:
When training with hard 0 or 1 labels, the model can become overconfident, leading to overfitting or poor generalization.

✅ Solution:
Label smoothing replaces the hard labels 0 and 1 with softer values like 0.05 and 0.95.