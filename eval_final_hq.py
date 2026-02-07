import torch
import torchvision
import os
import json
import numpy as np
from tqdm import tqdm
import trimesh
import io
from PIL import Image
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from utils.general_utils import PILtoTorch
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

def evaluate_with_cpu_mesh_gt(model_path, iteration, json_path, mesh_path, output_dir, num_samples=5):
    # 1. 加载 3DGS 模型
    gaussians = GaussianModel(DummyArgs())
    ply_path = os.path.join(model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
    gaussians.load_ply(ply_path)
    
    # 2. 加载位姿
    with open(json_path, 'r') as f:
        data = json.load(f)
        all_w2cs = np.array(data['w2cs_matrices'] if isinstance(data, dict) else data)
    sampled_w2cs = all_w2cs[:num_samples]

    # 3. 获取相机内参
    with open(os.path.join(model_path, "cameras.json"), 'r') as f:
        cam_info = json.load(f)[0]
    W, H = cam_info['width'], cam_info['height']
    fx, fy = cam_info['fx'], cam_info['fy']
    fovx = 2 * np.arctan(W / (2 * fx))
    fovy = 2 * np.arctan(H / (2 * fy))

    # 4. 加载 Mesh 并进行 CPU 软渲染生成高质量 GT
    print(f"Loading mesh from {mesh_path} for CPU rendering...")
    mesh = trimesh.load(mesh_path)
    
    gt_dir = os.path.join(output_dir, "gt_cpu")
    render_dir = os.path.join(output_dir, "renders")
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(render_dir, exist_ok=True)

    metrics = {"psnr": [], "ssim": [], "lpips": []}
    lpips_model = LPIPS(net_type='alex').cuda()
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    print(f"Generating High-Quality GT and Rendering {num_samples} frames...")
    for i in range(num_samples):
        w2c = sampled_w2cs[i]
        
        # --- A. CPU 软渲染 GT ---
        scene = mesh.scene()
        # 转换坐标系: OpenCV -> Trimesh
        c2w = np.linalg.inv(w2c)
        flip_yz = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
        scene.camera_transform = c2w @ flip_yz
        scene.camera.resolution = [W, H]
        scene.camera.focal = [fx, fy]
        
        gt_data = scene.save_image()
        gt_pil = Image.open(io.BytesIO(gt_data)).convert("RGB")
        gt_pil.save(os.path.join(gt_dir, f"{i:04d}.png"))
        gt_tensor = PILtoTorch(gt_pil, (W, H)).unsqueeze(0).cuda()

        # --- B. 3DGS 渲染 ---
        cam = Camera(colmap_id=i, R=w2c[:3, :3].transpose(), T=w2c[:3, 3], 
                     FoVx=fovx, FoVy=fovy, image=torch.zeros((3, H, W)), 
                     gt_alpha_mask=None, image_name=f"eval_{i}", uid=i, fid=0)
        render_pkg = render(cam, gaussians, DummyPipe(), background)
        rendered_img = render_pkg["render"].unsqueeze(0)
        torchvision.utils.save_image(rendered_img.squeeze(0), os.path.join(render_dir, f"{i:04d}.png"))

        # --- C. 计算指标 ---
        p = psnr(rendered_img, gt_tensor).item()
        s = ssim(rendered_img, gt_tensor).item()
        l = lpips_model(rendered_img, gt_tensor).item()
        
        metrics["psnr"].append(p)
        metrics["ssim"].append(s)
        metrics["lpips"].append(l)
        print(f"Frame {i:04d} | PSNR: {p:.2f}dB | SSIM: {s:.4f}")

    avg_psnr = np.mean(metrics["psnr"])
    print(f"\n✅ Final Average PSNR (HQ Mesh GT): {avg_psnr:.2f}dB")

if __name__ == "__main__":
    evaluate_with_cpu_mesh_gt(
        model_path="/workspace_fs/guidedvd-3dgs/output/replica_guidedvd_office2_0203_1509/office_2/Sequence_2",
        iteration=10000,
        json_path="/workspace_fs/robot_walk_2/w2cs.json",
        mesh_path="/workspace_fs/guidedvd-3dgs/dataset/Replica/office_2/mesh.ply",
        output_dir="/workspace_fs/guidedvd-3dgs/final_eval_hq",
        num_samples=5
    )
