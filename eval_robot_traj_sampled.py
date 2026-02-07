import torch
import os
import json
import numpy as np
from tqdm import tqdm
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
import torchvision
from utils.loss_utils import psnr, ssim
from lpipsPyTorch import lpips
import cv2
from PIL import Image

class DummyArgs:
    def __init__(self):
        self.sh_degree = 3

class DummyPipe:
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.use_confidence = False

def evaluate_sampled_robot_traj(model_path, iteration, json_path, gt_dir, output_dir, sampled_indices):
    """
    对采样的 100 个机器人视角计算 PSNR/SSIM/LPIPS。
    """
    # 1. 加载内参
    cameras_json_path = os.path.join(model_path, "cameras.json")
    with open(cameras_json_path, 'r') as f:
        cam_info = json.load(f)[0]
    
    W, H = cam_info['width'], cam_info['height']
    fx, fy = cam_info['fx'], cam_info['fy']
    fovx = 2 * np.arctan(W / (2 * fx))
    fovy = 2 * np.arctan(H / (2 * fy))

    # 2. 加载模型
    gaussians = GaussianModel(DummyArgs())
    ply_path = os.path.join(model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
    gaussians.load_ply(ply_path)
    pipe = DummyPipe()
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    # 3. 加载采样位姿
    with open(json_path, 'r') as f:
        all_w2cs = np.array(json.load(f)['w2cs_matrices'])
    
    render_dir = os.path.join(output_dir, "renders_sampled")
    os.makedirs(render_dir, exist_ok=True)

    metrics = {"psnr": [], "ssim": [], "lpips": []}

    print(f"Evaluating {len(sampled_indices)} sampled robot views...")
    for idx in tqdm(sampled_indices):
        w2c = all_w2cs[idx]
        R = w2c[:3, :3].transpose()
        T = w2c[:3, 3]
        
        # 渲染
        cam = Camera(colmap_id=idx, R=R, T=T, FoVx=fovx, FoVy=fovy, 
                     image=torch.zeros((3, H, W)), 
                     gt_alpha_mask=None, image_name=f"robot_{idx}", 
                     uid=idx, fid=0, data_device="cuda")
        
        render_pkg = render(cam, gaussians, pipe, background)
        rendered_img = render_pkg["render"] # [3, H, W], float32, 0-1
        
        # 保存渲染图
        torchvision.utils.save_image(rendered_img, os.path.join(render_dir, f"{idx:04d}.png"))

        # 加载 GT 图
        gt_path = os.path.join(gt_dir, f"{idx:04d}.png")
        if not os.path.exists(gt_path):
            print(f"Warning: GT not found for index {idx}")
            continue
            
        gt_img = Image.open(gt_path).convert('RGB')
        gt_tensor = torch.from_numpy(np.array(gt_img)).permute(2, 0, 1).float().cuda() / 255.0
        
        # 确保尺寸一致
        if gt_tensor.shape[1:] != rendered_img.shape[1:]:
            gt_tensor = torch.nn.functional.interpolate(gt_tensor.unsqueeze(0), size=(H, W), mode='bilinear').squeeze(0)

        # 计算指标
        psnr_val = psnr(rendered_img, gt_tensor)
        if psnr_val.numel() > 1:
            psnr_val = psnr_val.mean()
        metrics["psnr"].append(psnr_val.item())
        
        ssim_val = ssim(rendered_img.unsqueeze(0), gt_tensor.unsqueeze(0))
        metrics["ssim"].append(ssim_val.item())
        
        # LPIPS 需要 [1, 3, H, W]
        lpips_val = lpips(rendered_img.unsqueeze(0), gt_tensor.unsqueeze(0), net_type='vgg')
        metrics["lpips"].append(lpips_val.item())

    # 汇总结果
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    print("\n" + "="*30)
    print(f"Robot Path Metrics (Sampled 100):")
    for k, v in avg_metrics.items():
        print(f"  {k.upper()}: {v:.4f}")
    print("="*30)

    with open(os.path.join(output_dir, "robot_path_results.json"), "w") as f:
        json.dump({"avg": avg_metrics, "per_view": metrics}, f, indent=4)

if __name__ == "__main__":
    model_path = "/workspace_fs/guidedvd-3dgs/output/replica_guidedvd_office2_0203_1509/office_2/Sequence_2"
    iteration = 10000
    json_path = "/workspace_fs/guidedvd-3dgs/w2cs_ig.json"
    gt_dir = "/workspace_fs/guidedvd-3dgs/baseline_result/gt"
    output_dir = "/workspace_fs/guidedvd-3dgs/baseline_result"
    
    # 获取之前脚本生成的采样索引
    total_poses = 1199
    sampled_indices = np.linspace(0, total_poses - 1, 100, dtype=int)
    
    evaluate_sampled_robot_traj(model_path, iteration, json_path, gt_dir, output_dir, sampled_indices)
