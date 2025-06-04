
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# YOLO11 object detection model with P3-P5 outputs. For Usage examples see https://docs.ultralytics.com/tasks/detect

# # Parameters
# nc: 80 # number of classes
# scales: # model compound scaling constants, i.e. 'model=yolo11n.yaml' will call yolo11.yaml with scale 'n'
#   # [depth, width, max_channels]
#   n: [0.50, 0.25, 1024] # summary: 319 layers, 2624080 parameters, 2624064 gradients, 6.6 GFLOPs
#   s: [0.50, 0.50, 1024] # summary: 319 layers, 9458752 parameters, 9458736 gradients, 21.7 GFLOPs
#   m: [0.50, 1.00, 512] # summary: 409 layers, 20114688 parameters, 20114672 gradients, 68.5 GFLOPs
#   l: [1.00, 1.00, 512] # summary: 631 layers, 25372160 parameters, 25372144 gradients, 87.6 GFLOPs
#   x: [1.00, 1.50, 512] # summary: 631 layers, 56966176 parameters, 56966160 gradients, 196.0 GFLOPs

# # | Layer Index | Output Shape       | Notes                                   |
# # | ----------- | ------------------ | --------------------------------------- |
# # | 0–1         | 320x320, 160x160   | Basic Conv downsampling                 |
# # | 3 (P3)x8    | 80x80              | Medium-resolution feature               |
# # | 5 (P4)x16   | 40x40              | Lower-resolution feature                |
# # | 7 (P5)x32   | 20x20              | Deepest feature, used for small objects |
# # | 9–10        | SPPF and attention | Feature refinement                      |

# # YOLO11n backbone
# backbone:
#   # [from, repeats, module, args]
#            #Conv(out_channels, kernel_size, stride)
#   - [-1, 1, Conv, [64, 3, 2]]         # Layer 0:# 0-P1/2
#   - [-1, 1, Conv, [128, 3, 2]]        # Layer 1:# 1-P2/4
#   - [-1, 2, C3k2, [256, False, 0.25]] # Layer 2:
#   - [-1, 1, Conv, [256, 3, 2]]        # Layer 3: [2-3]      3-P3/8  resnet50 (C2)
#   - [-1, 2, C3k2, [512, False, 0.25]] # Layer 4:  
#   - [-1, 1, Conv, [512, 3, 2]]        # Layer 5: 
#   - [-1, 2, C3k2, [512, True]]        # Layer 6: [4-6]   P4/16 resnet50 (C3)
#   - [-1, 1, Conv, [1024, 3, 2]]       # Layer 7:#         
#   - [-1, 2, C3k2, [1024, True]]       # Layer 8: [7-8]    P5/32 resnet50 (C5)
#   - [-1, 1, SPPF, [1024, 5]] # 9      # Layer 9:
#   - [-1, 2, C2PSA, [1024]] # 10       # Layer 10:    Backbone P5 (20×20 → 8×8) 

# | Layer | Definition                                       | Output Shape / Comment                      |
# |-------|--------------------------------------------------|----------------------------------------------|

# | 14    | [-1, 1, nn.Upsample, [None, 2, "nearest"]]       | [B, 512, 64, 64] (Upsample P4)               |
# | 15    | [[-1, 4], 1, Concat, [1]]                        | [B, 512 + 256 = 768, 64, 64] (cat P3)        |
# | 16    | [-1, 2, C3k2, [256, False]]                      | [B, 256, 64, 64] (P3/8-small)                |
# | 17    | [-1, 1, Conv, [256, 3, 2]]                       | [B, 256, 32, 32] (Downsample to P4 scale)    |
# | 18    | [[-1, 13], 1, Concat, [1]]                       | [B, 256 + 512 = 768, 32, 32] (cat head P4)   |
# | 19    | [-1, 2, C3k2, [512, False]]                      | [B, 512, 32, 32] (P4/16-medium)              |
# | 20    | [-1, 1, Conv, [512, 3, 2]]                       | [B, 512, 16, 16] (Downsample to P5 scale)    |
# | 21    | [[-1, 10], 1, Concat, [1]]                       | [B, 512 + 1024 = 1536, 16, 16] (cat head P5) |
# | 22    | [-1, 2, C3k2, [1024, True]]                      | [B, 1024, 16, 16] (P5/32-large)              |
# | 23    | [[16, 19, 22], 1, YOLOEDetect, [nc, 512, True]]  | Detect head for P3 (64×64), P4 (32×32), P5 (16×16) |


# # YOLO11n head
# head:
#   - [-1, 1, nn.Upsample, [None, 2, "nearest"]]        # 11           [B, 1024, 16 → 32, 16 → 32] (Upsample P5) 
#   - [[-1, 6], 1, Concat, [1]] # cat backbone P4       # 12  [-1, 6] (cat P4)  1024 + 512 = [1536, 32, 32] 
#   - [-1, 2, C3k2, [512, False]] # 13                  # 13               512, 32 ,32     (refine P4 features) 

#   - [-1, 1, nn.Upsample, [None, 2, "nearest"]]        # 14 512 64 x 64  
#   - [[-1, 4], 1, Concat, [1]]                         # 15 [-1, 4] cat backbone P3 (512 + 256) = [768 x 64 x 64]
#   - [-1, 2, C3k2, [256, False]]                       # 16 (P3/8-small)  [256 x 64 x 64]  (refine p3 features) 

#   - [-1, 1, Conv, [256, 3, 2]]                        # 17   [256 = channel, 3 = Kernel size (3×3 convolution window), 2 = stride]   [B, 256, 32, 32]
#   - [[-1, 13], 1, Concat, [1]] # cat head P4          # 18    512 + 256[from 13 - P4]  [768 x 32 x 32]
#   - [-1, 2, C3k2, [512, False]]                       # 19   [512 x 32 x 32]  (P4/16-medium) 

#   - [-1, 1, Conv, [512, 3, 2]]                        # 20   [512 x 16 x 16]  
#   - [[-1, 10], 1, Concat, [1]] # cat head P5          # 21   512 + 1024(C2PSA) [1536 X 12 X 12]
#   - [-1, 2, C3k2, [1024, True]] # 22                  # 21   [1024 X 12 X 12]  (P5/32-large)

#   - [[16, 19, 22], 1, YOLOEDetect, [nc, 512, True]] # Detect(P3, P4, P5)

# nc: 1  # number of classes
# backbone:
# # (f, n, m, args)
# # from number module args
# #  i      f       t
# # m_.i, m_.f, m_.type 
#   - [-1, 1, Timm, [512, 'efficientnet_b5', True, True, 0, True]]  # Layer 0: returns list of block outputs
#   - [0, 1, Index, [64, 2]]     # Layer 1: P3 = Block 2 output (32x32)
#   - [0, 1, Index, [176, 3]]    # Layer 2: P4 = Block 4 output (16x16)
#   - [0, 1, Index, [512, 4]]    # Layer 3: P5 = Block 6 output (8x8)
#   - [-1, 1, SPPF, [512, 5]]    # Layer 4: SPPF on P5 (8x8) to enhance receptive field

# head:
#   # 上采样/拼接/检测头，通道数建议与 backbone 输出保持一致
#   - [-1, 1, nn.Upsample, [None, 2, 'nearest']]            # 5,  #scale_factor=2 SPPF上采样 (8->16) Upsamples the output from Layer 4, which has 512 channels from SPPF.
#   - [[-1, 2], 1, Concat, [1]]                             # 6, 拼接16x16的两个特征 [-1, 2] =>  prev + P4 =>  512 + 176 = 688
#   - [-1, 3, C2f, [176]]                                   # 7, 通道数和P4对齐  C2f(in_channels=688, out_channels=176) Tensor shape: [B, 688, 16, 16] => [B, 176, 16, 16]

#   - [-1, 1, nn.Upsample, [None, 2, 'nearest']]            # 8, 上采样 (16->32)   [B, 176, 32, 32]
#   - [[-1, 1], 1, Concat, [1]]                             # 9, 拼接32x32的两个特征  [-1, 1] => prev + P3 =>  176 + 64 => [B, 240, 32, 32]
#   - [-1, 3, C2f, [64]]                                    # 10, 通道数和P3对齐    (P3/8-small) [B, 64, 32, 32] 

#   - [-1, 1, Conv, [64, 3, 2]]                             # 11, 下采样 (32->16)   [B, 64, 16, 16]
#   - [[-1, 7], 1, Concat, [1]]                             # 12, 拼接16x16的两个特征     176 + 64 [B, 240, 16, 16]
#   - [-1, 3, C2f, [176]]                                   # 13                        [B, 176, 16, 16]

#   - [-1, 1, Conv, [176, 3, 2]]                            # 14, 下采样 (16->8)          [B, 176, 8, 8]
#   - [[-1, 4], 1, Concat, [1]]                             # 15, 拼接8x8的两个特征       176 + 512 = 688  [B, 688, 8, 8]
#   - [-1, 3, C2f, [512]]                                   # 16                            [B, 512, 8, 8]

#   - [[10, 13, 16], 1, Detect, [nc]]                       # 17, 检测头, 多尺度


# Feature 0: torch.Size([1, 24, 228, 228])
# Feature 1: torch.Size([1, 40, 114, 114])
# Feature 2: torch.Size([1, 64, 57, 57])
# Feature 3: torch.Size([1, 176, 29, 29])
# Feature 4: torch.Size([1, 512, 15, 15])


import os
from ultralytics import YOLO
import torch
local_dev =  "/workspace/BYU/notebooks" if "WANDB_API_KEY" in os.environ else "C:/Users/Freedomkwok2022/ML_Learn/BYU/notebooks"
yolo_dataset_dir = os.path.join(local_dev, 'yolo_dataset')
yolo_weights_dir = os.path.join(local_dev, 'yolo_weights')
yolo_models_dir = os.path.join(local_dev, 'models')

img = torch.rand(1, 3, 640, 640)

efficientnet_b5_yolo_config_path = os.path.join(yolo_models_dir, "b5.yaml")
# 48,691,139
model = YOLO(efficientnet_b5_yolo_config_path, verbose=False)
print(model.info())
feature_outputs = {}

def hook_fn(name):
    def fn(module, input, output):
        if isinstance(output, torch.Tensor):
            numel = output.numel()
            print("numel", numel)
            if numel > 2_000_000_000:
                raise ValueError(f"[{name}] Output too large for 32-bit indexing: numel={numel}")
        feature_outputs[name] = output
    return fn

# Register hooks (adjust indices to match your model architecture)
model.model.model[11].register_forward_hook(hook_fn("P3"))
model.model.model[15].register_forward_hook(hook_fn("P4"))
model.model.model[19].register_forward_hook(hook_fn("P5"))

# Run inference
with torch.no_grad():
    _ = model(img)

# Print captured features
for name, feat in feature_outputs.items():
    print(f"{name} output shape: {feat.shape}")

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
def hook_fn(name):
    def fn(module, input, output):
        feature_outputs[name] = output
    return fn

# Register hooks on YOLOv8 internal layers — adjust indices as needed
model.model.model[11].register_forward_hook(hook_fn("P3"))
model.model.model[15].register_forward_hook(hook_fn("P4"))
model.model.model[19].register_forward_hook(hook_fn("P5"))

# Run inference to trigger hooks
with torch.no_grad():
    _ = model(img)

# Print captured feature shapes
for name, feat in feature_outputs.items():
    print(f"{name} output shape: {feat.shape}")