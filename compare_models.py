import os
import torch
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as tf
from argparse import ArgumentParser

from gaussian_renderer import render
from scene import GaussianModel
from utils.graphics_utils import getWorld2View2, focal2fov
from utils.loss_utils import ssim
from utils.image_utils import psnr
from lpipsPyTorch import lpips

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform, camera_center):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        self.camera_center = camera_center
        self.data_device = "cuda"

def get_projection_matrix(znear, zfar, fovX, fovY):
    tanHalfFovY = np.tan(fovY / 2)
    tanHalfFovX = np.tan(fovX / 2)
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right
    P = torch.zeros(4, 4)
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def load_gaussians(model_path, iteration):
    args_gs = type('obj', (object,), {'sh_degree': 3})()
    gaussians = GaussianModel(args_gs)
    ply_path = os.path.join(model_path, f"point_cloud/iteration_{iteration}/point_cloud.ply")
    gaussians.load_ply(ply_path)
    return gaussians

def compare_models(baseline_path, ours_path, iteration, w2cs_path, gt_images_path, num_frames=200):
    # Load Gaussians for both
    print("Loading models...")
    gaussians_base = load_gaussians(baseline_path, iteration)
    gaussians_ours = load_gaussians(ours_path, iteration)
    
    # Load Poses
    with open(w2cs_path, 'r') as f:
        data = json.load(f)
        w2cs = np.array(data['w2cs_matrices'][:num_frames])

    # Camera settings
    width, height = 518, 518
    focal = 320.0
    fovx = focal2fov(focal, width)
    fovy = focal2fov(focal, height)
    znear, zfar = 0.01, 100.0
    projection_matrix = get_projection_matrix(znear, zfar, fovx, fovy).transpose(0, 1).cuda()
    
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    pipeline_params = type('obj', (object,), {
        'convert_SHs_python': False, 
        'compute_cov3D_python': False, 
        'debug': False,
        'use_confidence': False,
        'use_color': True
    })()

    results = []
    
    output_dir = "comparison_results"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Comparing {num_frames} frames...")
    
    for i in tqdm(range(num_frames)):
        w2c = w2cs[i]
        R = w2c[:3, :3].T
        T = w2c[:3, 3]
        world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).cuda()
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]
        cam = MiniCam(width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform, camera_center)
        
        # Load GT
        gt_path = os.path.join(gt_images_path, f"{i:06d}.png")
        gt_image = Image.open(gt_path).convert("RGB")
        gt_tensor = tf.to_tensor(gt_image).cuda()

        # Render Baseline
        render_base = render(cam, gaussians_base, pipeline_params, bg_color)["render"]
        psnr_base = psnr(torch.clamp(render_base, 0, 1), gt_tensor).mean().item()
        
        # Render Ours
        render_ours = render(cam, gaussians_ours, pipeline_params, bg_color)["render"]
        psnr_ours = psnr(torch.clamp(render_ours, 0, 1), gt_tensor).mean().item()
        
        diff = psnr_ours - psnr_base
        results.append({
            'frame': i,
            'psnr_base': psnr_base,
            'psnr_ours': psnr_ours,
            'diff': diff
        })

    # Sort by diff descending to find best improvements
    results.sort(key=lambda x: x['diff'], reverse=True)
    
    print("\n" + "="*50)
    print(f"{'Frame':<10} | {'Baseline PSNR':<15} | {'Ours PSNR':<15} | {'Improvement':<10}")
    print("-" * 55)
    for res in results[:10]:
        print(f"{res['frame']:<10} | {res['psnr_base']:<15.4f} | {res['psnr_ours']:<15.4f} | {res['diff']:<10.4f}")
    print("="*50)

    # Save the top 3 improved frames for visual inspection
    for res in results[:3]:
        i = res['frame']
        # Re-render to save
        w2c = w2cs[i]
        R = w2c[:3, :3].T
        T = w2c[:3, 3]
        world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).cuda()
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]
        cam = MiniCam(width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform, camera_center)
        
        r_base = torch.clamp(render(cam, gaussians_base, pipeline_params, bg_color)["render"], 0, 1)
        r_ours = torch.clamp(render(cam, gaussians_ours, pipeline_params, bg_color)["render"], 0, 1)
        gt_path = os.path.join(gt_images_path, f"{i:06d}.png")
        gt = tf.to_tensor(Image.open(gt_path).convert("RGB")).cuda()
        
        combined = torch.cat([r_base, r_ours, gt], dim=2)
        import torchvision
        torchvision.utils.save_image(combined, f"{output_dir}/best_improvement_frame_{i:06d}_diff_{res['diff']:.2f}.png")
    
    print(f"Top 3 improved frames saved to {output_dir}/")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--baseline", type=str, required=True)
    parser.add_argument("--ours", type=str, required=True)
    parser.add_argument("--iteration", type=int, default=10000)
    parser.add_argument("--w2cs", type=str, required=True)
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--num_frames", type=int, default=200)
    args = parser.parse_args()
    
    compare_models(args.baseline, args.ours, args.iteration, args.w2cs, args.gt_dir, args.num_frames)
