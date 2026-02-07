import torch
import os
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import ToTensor
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from utils.loss_utils import psnr, ssim
from lpipsPyTorch import LPIPS

class DummyArgs:
    def __init__(self): self.sh_degree = 3
class DummyPipe:
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.use_confidence = False

def evaluate_fixed_frames(model_path, iteration, source_path, test_indices):
    # 1. Load Gaussians
    gaussians = GaussianModel(DummyArgs())
    ply_path = os.path.join(model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
    gaussians.load_ply(ply_path)

    # 2. Load Camera Info (Replica specific)
    with open(os.path.join(model_path, "cameras.json"), 'r') as f:
        cam_infos = json.load(f)
    
    # Create a mapping from img_name to cam_info
    cam_dict = {c['img_name']: c for c in cam_infos}

    lpips_model = LPIPS(net_type='alex').cuda()
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    
    results = {}
    
    print(f"Evaluating {len(test_indices)} fixed frames with real GT...")
    for idx in tqdm(test_indices):
        img_name = f"rgb_{idx}"
        if img_name not in cam_dict:
            # Try to find if the index exists in any img_name
            found = False
            for k in cam_dict.keys():
                if f"_{idx}" in k or k == str(idx):
                    img_name = k
                    found = True
                    break
            if not found:
                print(f"Warning: Camera info for {idx} not found, skipping.")
                continue
        
        c = cam_dict[img_name]
        W, H = c['width'], c['height']
        fx, fy = c['fx'], c['fy']
        fovx = 2 * np.arctan(W / (2 * fx))
        fovy = 2 * np.arctan(H / (2 * fy))
        
        # Build Camera object
        # The cameras.json for this model uses 'position' and 'rotation'
        R = np.array(c['rotation'])
        T = np.array(c['position'])
        
        # 3DGS Camera expects R to be the rotation matrix from world to camera
        # and T to be the translation vector from world to camera.
        # However, the 'render' function in this repo might expect them differently.
        # Based on typical 3DGS implementations:
        # cam = Camera(..., R=R, T=T, ...)
        
        cam = Camera(colmap_id=idx, R=R, T=T, 
                     FoVx=fovx, FoVy=fovy, image=torch.zeros((3, H, W)), 
                     gt_alpha_mask=None, image_name=img_name, 
                     uid=idx, fid=0)
        
        # Render
        render_pkg = render(cam, gaussians, DummyPipe(), background)
        rendered_img = render_pkg["render"]
        
        # Load Real GT
        gt_path = os.path.join(source_path, "rgb", f"rgb_{idx}.png")
        if not os.path.exists(gt_path):
            gt_path = os.path.join(source_path, "images", f"rgb_{idx}.png")
            
        gt_image = Image.open(gt_path)
        gt_tensor = ToTensor()(gt_image).cuda()

        # Calculate Metrics
        p = psnr(rendered_img.unsqueeze(0), gt_tensor.unsqueeze(0)).item()
        s = ssim(rendered_img.unsqueeze(0), gt_tensor.unsqueeze(0)).item()
        l = lpips_model(rendered_img.unsqueeze(0), gt_tensor.unsqueeze(0)).item()
        
        results[idx] = {"psnr": p, "ssim": s, "lpips": l}

    # Summary
    avg_psnr = np.mean([v['psnr'] for v in results.values()])
    avg_ssim = np.mean([v['ssim'] for v in results.values()])
    avg_lpips = np.mean([v['lpips'] for v in results.values()])
    
    print(f"\n--- Final Results for Fixed Indices ---")
    print(f"Average PSNR: {avg_psnr:.4f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average LPIPS: {avg_lpips:.4f}")
    
    with open("baseline_fixed_results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    indices = [150, 151, 152, 138, 136, 137, 139, 149, 153, 135, 134, 140, 148, 154, 133, 141, 147, 155, 132, 142]
    evaluate_fixed_frames(
        model_path="/workspace_fs/guidedvd-3dgs/output/replica_guidedvd_office2_0203_1509/office_2/Sequence_2",
        iteration=10000,
        source_path="/workspace_fs/guidedvd-3dgs/dataset/Replica/office_2/Sequence_2",
        test_indices=indices
    )
