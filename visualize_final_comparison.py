import os
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def create_comparison_grid(gt_dir, render_baseline_dir, render_ours_dir, output_path, indices):
    images_per_row = 3 # GT, Baseline, Ours
    num_rows = len(indices)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()

    grid_cells = []
    
    for idx in indices:
        # 1. Load GT
        gt_path = os.path.join(gt_dir, f"rgb_{idx}.png")
        # 2. Load Baseline
        base_path = os.path.join(render_baseline_dir, f"rgb_{idx}.png")
        # 3. Load Ours
        ours_path = os.path.join(render_ours_dir, f"rgb_{idx}.png")
        
        row_imgs = []
        for p, label in [(gt_path, "GT"), (base_path, "Baseline"), (ours_path, "Ours (Task-Specific)")]:
            if os.path.exists(p):
                img = Image.open(p).convert("RGB")
                # Resize if needed (assuming 640x480)
                img = img.resize((640, 480))
                
                # Draw Label
                draw = ImageDraw.Draw(img)
                draw.text((10, 10), label, fill="red", font=font)
                row_imgs.append(img)
            else:
                # Placeholder
                row_imgs.append(Image.new("RGB", (640, 480), (0, 0, 0)))
        
        grid_cells.append(row_imgs)

    # Combine into one big image
    W, H = 640, 480
    combined = Image.new("RGB", (W * 3, H * num_rows))
    
    for r_idx, row in enumerate(grid_cells):
        for c_idx, img in enumerate(row):
            combined.paste(img, (c_idx * W, r_idx * H))
            
    combined.save(output_path)
    print(f"✅ Comparison grid saved to {output_path}")

if __name__ == "__main__":
    # 使用你关心的固定帧索引
    test_indices = [150, 151, 152, 138, 136] # 示例几个
    
    # 实际上 Baseline 只有特定的帧，我们需要找交集或者对应的帧
    # 之前我们跑 Baseline 得到的帧是: [194, 204, 214, 224, 234]
    indices_to_show = [194, 204, 214, 224, 234] 

    create_comparison_grid(
        gt_dir="/workspace_fs/guidedvd-3dgs/dataset/Replica/office_2/Sequence_2/rgb",
        render_baseline_dir="/workspace_fs/guidedvd-3dgs/output/replica_guidedvd_office2_0203_1509/office_2/Sequence_2/test/ours_10000/renders",
        render_ours_dir="/workspace_fs/guidedvd-3dgs/output/replica_guidedvd_office2_task_specific_fixed_test/images/test", # 确认这个路径
        output_path="/workspace_fs/guidedvd-3dgs/baseline_vs_ours_vis/final_comparison.png",
        indices=indices_to_show
    )
