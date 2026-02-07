import torch
import torchvision
import os
import json
import numpy as np
from tqdm import tqdm
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from utils.general_utils import PILtoTorch
from utils.loss_utils import psnr, ssim
from lpipsPyTorch import LPIPS
from PIL import Image
import trimesh
import imageio
import cv2

class DummyArgs:
    def __init__(self): self.sh_degree = 3
class DummyPipe:
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.use_confidence = False

def generate_gt_manual_sequential(mesh_path, w2cs, output_dir, cam_info):
    print(f"Loading mesh from {mesh_path}...")
    tm = trimesh.load(mesh_path)
    vertices = tm.vertices
    vertex_colors = tm.visual.vertex_colors[:, :3] / 255.0

    gt_dir = os.path.join(output_dir, "gt_new")
    os.makedirs(gt_dir, exist_ok=True)

    W, H = cam_info['width'], cam_info['height']
    fx, fy = cam_info['fx'], cam_info['fy']
    cx, cy = W / 2, H / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    print(f"Projecting {len(w2cs)} sequential views manually...")
    for idx in tqdm(range(len(w2cs))):
        w2c = w2cs[idx]
        R_w2c = w2c[:3, :3]
        t_w2c = w2c[:3, 3]
        
        vertices_cam = (R_w2c @ vertices.T).T + t_w2c
        vertices_proj_h = (K @ vertices_cam.T).T
        depths = vertices_proj_h[:, 2]
        valid_mask = depths > 1e-6
        
        pixel_coords = vertices_proj_h[valid_mask, :2] / depths[valid_mask, None]
        colors = vertex_colors[valid_mask]
        x_coords, y_coords = pixel_coords[:, 0], pixel_coords[:, 1]
        in_bounds_mask = (x_coords >= 0) & (x_coords < W) & (y_coords >= 0) & (y_coords < H)
        
        final_pixel_coords = pixel_coords[in_bounds_mask].astype(int)
        final_colors = (colors[in_bounds_mask] * 255).astype(np.uint8)
        final_depths = depths[valid_mask][in_bounds_mask]

        image = np.zeros((H, W, 3), dtype=np.uint8)
        depth_buffer = np.full((H, W), np.inf, dtype=np.float32)
        sort_indices = np.argsort(final_depths)[::-1]
        
        for p_idx in sort_indices:
            px, py = final_pixel_coords[p_idx]
            if final_depths[p_idx] < depth_buffer[py, px]:
                image[py, px] = final_colors[p_idx]
                depth_buffer[py, px] = final_depths[p_idx]

        imageio.imwrite(os.path.join(gt_dir, f"gt_{idx:04d}.png"), image)
    return gt_dir

def evaluate_new_trajectory(model_path, iteration, json_path, mesh_path, output_dir, num_samples=50):
    gaussians = GaussianModel(DummyArgs())
    ply_path = os.path.join(model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
    gaussians.load_ply(ply_path)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
        all_w2cs = np.array(data['w2cs_matrices'] if isinstance(data, dict) else data)
    
    sampled_indices = np.arange(min(num_samples, len(all_w2cs)))
    sampled_w2cs = all_w2cs[sampled_indices]

    with open(os.path.join(model_path, "cameras.json"), 'r') as f:
        cam_info = json.load(f)[0]
    W, H = cam_info['width'], cam_info['height']
    fovx, fovy = 2 * np.arctan(W / (2 * cam_info['fx'])), 2 * np.arctan(H / (2 * cam_info['fy']))

    gt_dir = os.path.join(output_dir, "gt_new")
    if not os.path.exists(gt_dir) or len(os.listdir(gt_dir)) < len(sampled_indices):
        gt_dir = generate_gt_manual_sequential(mesh_path, sampled_w2cs, output_dir, cam_info)

    render_dir = os.path.join(output_dir, "renders_new")
    os.makedirs(render_dir, exist_ok=True)

    metrics = {"psnr": [], "ssim": [], "lpips": []}
    lpips_model = LPIPS(net_type='alex').cuda()
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    print(f"Rendering and evaluating {len(sampled_indices)} new robot views...")
    for i, idx in enumerate(tqdm(sampled_indices)):
        w2c = all_w2cs[idx]
        cam = Camera(colmap_id=idx, R=w2c[:3, :3].transpose(), T=w2c[:3, 3], 
                     FoVx=fovx, FoVy=fovy, image=torch.zeros((3, H, W)), 
                     gt_alpha_mask=None, image_name=f"robot_debug_{idx}", 
                     uid=idx, fid=0)
        
        render_pkg = render(cam, gaussians, DummyPipe(), background)
        rendered_img = render_pkg["render"].unsqueeze(0)
        
        gt_image_path = os.path.join(gt_dir, f"gt_{idx:04d}.png")
        gt_pil = Image.open(gt_image_path)
        gt_tensor = PILtoTorch(gt_pil, (W, H)).unsqueeze(0).cuda()

        p = psnr(rendered_img, gt_tensor).item()
        s = ssim(rendered_img, gt_tensor).item()
        l = lpips_model(rendered_img, gt_tensor).item()
        
        metrics["psnr"].append(p)
        metrics["ssim"].append(s)
        metrics["lpips"].append(l)
        
        torchvision.utils.save_image(rendered_img.squeeze(0), os.path.join(render_dir, f"render_{idx:04d}.png"))

    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    print(f"\n--- Metrics for first {len(sampled_indices)} frames ---")
    for k, v in avg_metrics.items(): print(f"Average {k.upper()}: {v:.4f}")

    save_data = {
        "avg_metrics": avg_metrics,
        "per_frame_psnr": metrics["psnr"],
        "per_frame_ssim": metrics["ssim"],
        "per_frame_lpips": metrics["lpips"]
    }
    with open(os.path.join(output_dir, "final_results.json"), "w") as f:
        json.dump(save_data, f, indent=4)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--iteration", type=int, default=10000)
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--mesh_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=50)
    args = parser.parse_args()

    evaluate_new_trajectory(args.model_path, args.iteration, args.json_path, args.mesh_path, args.output_dir, args.num_samples)
