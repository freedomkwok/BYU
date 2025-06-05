	1.	Box大小应该用比例值还是固定值32-36？
	•	A. 使用固定值
	•	B. 使用10%的比例（因为每张图片大小不同，固定值可能不准确）✅（我使用）
	2.	测试图片像素过大时，YOLO（640）本身会自动缩放，但特征可能在缩放过程中丢失。
	•	A. 使用scale参数即可解决该问题
	•	B. 在预处理阶段将过大的图片切分后再交由模型（imgsz=640）处理 ✅（我使用）
也可以训练多个不同尺寸的模型 imgsz = [320, 640, 960]，预处理根据图片大小选择对应模型
	3.	目标过大或过小如何处理？
	•	过大：使用ATT方法预处理，将过大的图片分别按4个scale [0.8, 0.6, 0.5, 0.4] 缩放到640×640，然后输入模型；训练时也将这些缩放图与原图一并训练，推理后再映射回原图位置 ✅（我使用）
	•	过小：不处理？模型本身可以检测小目标


1.	Should bounding box size be defined using absolute values (like 32–36) or relative proportions?
	•	A. Use fixed values
	•	B. Use proportions (e.g., 10%) since image sizes vary and fixed sizes may be inaccurate ✅ (My current method)
	2.	What to do when test images are too large in resolution?
YOLO (with imgsz=640) automatically rescales them, but features might get lost during resizing.
	•	A. Use the scale parameter to handle this automatically
	•	B. Preprocess large images by splitting them and then feed them to a model trained at imgsz=640 ✅ (My current method)
Alternatively, train multiple models with different input sizes (e.g., imgsz = [320, 640, 960]) and dynamically select the model based on image resolution
	3.	How to handle targets that are too large or too small?
	•	Too large: Use an attention-based preprocessing strategy — resize overly large images to 640×640 using multiple downscale ratios like [0.8, 0.6, 0.5, 0.4], then feed to the model. Include both the scaled and original versions during training, and map predictions back to original coordinates after inference ✅ (My current method)
	•	Too small: No special handling — rely on the model’s built-in small-object detection capability