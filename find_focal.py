import numpy as np
from PIL import Image
import torch
import os
import json
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from utils.graphics_utils import focal2fov
from utils.image_utils import psnr
from argparse import Namespace

def find_best_focal(model_path, iteration, w2cs_path, gt_images_path):
    # Setup
    train_args = Namespace(sh_degree=3, data_device="cuda")
    gaussians = GaussianModel(train_args)
    ply_path = os.path.join(model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
    gaussians.load_ply(ply_path)
    
    with open(w2cs_path, 'r') as f:
        w2c = np.array(json.load(f)['w2cs_matrices'][0])
    
    gt_path = os.path.join(gt_images_path, "000000.png")
    gt = Image.open(gt_path).convert("RGB")
    width, height = gt.size
    gt_tensor = torch.from_numpy(np.array(gt)/255.0).permute(2, 0, 1).float().cuda()
    
    R = w2c[:3, :3].transpose()
    T = w2c[:3, 3]
    
    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, use_confidence=False, use_color=True)
    
    best_psnr = -1
    best_focal = -1
    
    print(f"Searching focal length for {width}x{height} image...")
    # Try a range of plausible focals
    for f in np.linspace(240, 280, 41):
        fovx = focal2fov(f, width)
        fovy = focal2fov(f, height)
        
        cam = Camera(colmap_id=0, R=R, T=T, FoVx=fovx, FoVy=fovy, 
                     image=torch.zeros((3, height, width)), gt_alpha_mask=None,
                     image_name="test", uid=0, fid=0)
        
        r = render(cam, gaussians, pipe, bg)["render"]
        p = psnr(torch.clamp(r, 0, 1), gt_tensor).mean().item()
        
        if p > best_psnr:
            best_psnr = p
            best_focal = f
            print(f"Focal {f:.1f} -> PSNR {p:.4f} (New Best)")
            
    print(f"\nConclusion: Best Focal is {best_focal:.1f} with PSNR {best_psnr:.4f}")

if __name__ == "__main__":
    find_best_focal(
        "/workspace_fs/robot_vdg/output/replica_office2_task_specific_0210_0213/ours_task_specific_fixed",
        10000,
        "/workspace_fs/robot_walk_2/w2cs.json",
        "/workspace_fs/robot_walk_2/gt_images"
    )
