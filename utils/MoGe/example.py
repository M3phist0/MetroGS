import cv2
import torch
# from moge.model.v1 import MoGeModel
from moge.model.v2 import MoGeModel # Let's try MoGe-2

device = torch.device("cuda")

# Load the model from huggingface hub (or load from local).
model = MoGeModel.from_pretrained("checkpoints/moge-2-vitl-normal.pt").to(device)                             

# Read the input image and convert to tensor (3, H, W) with RGB values normalized to [0, 1]
input_image = cv2.cvtColor(cv2.imread("/mnt/sharedisk/chenkehua/GS+Nerf/Grendel-GS/aaags/data/GauU_Scene/CUHK_UPPER_COLMAP/images/DJI_20231219115048_0084_Zenmuse-L1-mission.JPG"), cv2.COLOR_BGR2RGB)                       
input_image = torch.tensor(input_image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)    

# Infer 
output = model.infer(input_image)

depthmap = output['depth'].detach()
mask = output['mask'].detach()
depthmap[~mask] = 0

normalmap = output['normal'].detach()

import os
import numpy as np
os.makedirs('./tmp', exist_ok=True)
depth = depthmap.detach().cpu().numpy()
depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)
cv2.imwrite(os.path.join("./tmp/",  f"test2_depth.png"), depth_color)

normal = normalmap
normal = normal/(normal.norm(dim=-1, keepdim=True)+1.0e-8)
normal = normal.detach().cpu().numpy()
normal = ((normal+1) * 127.5).astype(np.uint8).clip(0, 255)
cv2.imwrite(os.path.join("./tmp/",  f"test2_normal.png"), normal)

"""
`output` has keys "points", "depth", "mask", "normal" (optional) and "intrinsics",
The maps are in the same size as the input image. 
{
    "points": (H, W, 3),    # point map in OpenCV camera coordinate system (x right, y down, z forward). For MoGe-2, the point map is in metric scale.
    "depth": (H, W),        # depth map
    "normal": (H, W, 3)     # normal map in OpenCV camera coordinate system. (available for MoGe-2-normal)
    "mask": (H, W),         # a binary mask for valid pixels. 
    "intrinsics": (3, 3),   # normalized camera intrinsics
}
"""