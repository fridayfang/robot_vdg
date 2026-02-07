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

def evaluate_available_test_frames(model_path, iteration, source_path):
    # 1. Load Gaussians
    gaussians = GaussianModel(DummyArgs())
    ply_path = os.path.join(model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
    gaussians.load_ply(ply_path)

    # 2. Load Camera Info
    with open(os.path.join(model_path, "cameras.json"), 'r') as f:
        cam_infos = json.load(f)
    
    # Filter for test frames (those NOT in the 6 training views)
    # Training views for office2_seq2: [244, 291, 436, 607, 760, 831]
    train_indices = [244, 291, 436, 607, 760, 831]
    
    test_cams = []
    for c in cam_infos:
        img_name = c['img_name']
        idx = int(img_name.split('_')[1])
        if idx not in train_indices:
            test_cams.append(c)
    
    # Sort by index and take first 20 for consistency
    test_cams.sort(key=lambda x: int(x['img_name'].split('_')[1]))
    test_cams = test_cams[:20]

    lpips_model = LPIPS(net_type='alex').cuda()
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    
    results = {}
    
    print(f"Evaluating {len(test_cams)} available test frames from Baseline model...")
    for c in tqdm(test_cams):
        img_name = c['img_name']
        idx = int(img_name.split('_')[1])
        
        W, H = c['width'], c['height']
        fx, fy = c['fx'], c['fy']
        fovx = 2 * np.arctan(W / (2 * fx))
        fovy = 2 * np.arctan(H / (2 * fy))
        
        # Build Camera object
        R = np.array(c['rotation'])
        T = np.array(c['position'])
        
        cam = Camera(colmap_id=idx, R=R, T=T, 
                     FoVx=fovx, FoVy=fovy, image=torch.zeros((3, H, W)), 
                     gt_alpha_mask=None, image_name=img_name, 
                     uid=idx, fid=0)
        
        # Render
        render_pkg = render(cam, gaussians, DummyPipe(), background)
        rendered_img = render_pkg["render"]
        
        # Load Real GT
        gt_path = os.path.join(source_path, "rgb", f"{img_name}.png")
        if not os.path.exists(gt_path):
            print(f"Warning: GT not found at {gt_path}, skipping.")
            continue
            
        gt_image = Image.open(gt_path)
        gt_tensor = ToTensor()(gt_image).cuda()

        # Calculate Metrics
        p = psnr(rendered_img.unsqueeze(0), gt_tensor.unsqueeze(0)).item()
        s = ssim(rendered_img.unsqueeze(0), gt_tensor.unsqueeze(0)).item()
        l = lpips_model(rendered_img.unsqueeze(0), gt_tensor.unsqueeze(0)).item()
        
        results[idx] = {"psnr": p, "ssim": s, "lpips": l}

    # Summary
    if not results:
        print("No results calculated.")
        return

    avg_psnr = np.mean([v['psnr'] for v in results.values()])
    avg_ssim = np.mean([v['ssim'] for v in results.values()])
    avg_lpips = np.mean([v['lpips'] for v in results.values()])
    
    print(f"\n--- Final Results for Baseline Available Test Frames ---")
    print(f"Indices evaluated: {list(results.keys())}")
    print(f"Average PSNR: {avg_psnr:.4f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average LPIPS: {avg_lpips:.4f}")
    
    with open("baseline_actual_test_results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    evaluate_available_test_frames(
        model_path="/workspace_fs/guidedvd-3dgs/output/replica_guidedvd_office2_0203_1509/office_2/Sequence_2",
        iteration=10000,
        source_path="/workspace_fs/guidedvd-3dgs/dataset/Replica/office_2/Sequence_2"
    )
