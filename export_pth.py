import torch
import torchvision
import os
import sys
from PIL import Image
import numpy as np

def export_pth_to_video(pth_path, output_dir):
    if not os.path.exists(pth_path):
        print(f"Error: File {pth_path} not found.")
        return

    print(f"Loading {pth_path}...")
    # pth 文件通常包含 {'img': tensor, 'c2w': tensor, ...} 或者直接是 tensor
    data = torch.load(pth_path, map_location='cpu')
    
    # 提取图像张量
    if isinstance(data, dict) and 'img' in data:
        frames = data['img'] # 预期形状 [N, 3, H, W] 或 [N, H, W, 3]
    else:
        frames = data

    if not isinstance(frames, torch.Tensor):
        print("Error: Could not find image tensor in pth file.")
        return

    # 确保形状是 [N, 3, H, W]
    if frames.dim() == 4:
        if frames.shape[-1] == 3: # [N, H, W, 3] -> [N, 3, H, W]
            frames = frames.permute(0, 3, 1, 2)
    else:
        print(f"Error: Unexpected tensor shape {frames.shape}")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Exporting {len(frames)} frames to {output_dir}...")
    for i, frame in enumerate(frames):
        # 归一化到 0-1 (ViewCrafter 输出通常在 [-1, 1] 或 [0, 1])
        if frame.min() < 0:
            frame = (frame + 1.0) / 2.0
        
        save_path = os.path.join(output_dir, f"frame_{i:03d}.png")
        torchvision.utils.save_image(frame, save_path)
    
    print(f"✅ Success! Frames saved in {output_dir}")

if __name__ == "__main__":
    # 默认转换我们刚才看到的那个文件
    default_pth = "output/replica_guidedvd_office2/office_2/Sequence_2/video_files_scale1/5/18.pth"
    target_pth = sys.argv[1] if len(sys.argv) > 1 else default_pth
    output_path = "output/debug_video_frames"
    
    export_pth_to_video(target_pth, output_path)
