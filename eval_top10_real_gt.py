import torch
import torchvision
import os
import json
import numpy as np
from tqdm import tqdm
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

def evaluate_top10_with_seq1_gt():
    model_path = "/workspace_fs/guidedvd-3dgs/output/replica_guidedvd_office2_0203_1509/office_2/Sequence_2"
    top10_json = "/workspace_fs/guidedvd-3dgs/top10_seq1_nearest.json"
    seq1_rgb_dir = "/workspace_fs/guidedvd-3dgs/dataset/Replica/office_2/Sequence_1/rgb"
    output_dir = "/workspace_fs/guidedvd-3dgs/eval_top10_seq1_gt"
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载 3DGS 模型
    gaussians = GaussianModel(DummyArgs())
    gaussians.load_ply(os.path.join(model_path, "point_cloud/iteration_10000/point_cloud.ply"))
    
    # 2. 加载 Top 10 信息
    with open(top10_json, 'r') as f:
        top10_data = json.load(f)

    # 3. 获取相机内参 (从训练配置读取)
    with open(os.path.join(model_path, "cameras.json"), 'r') as f:
        cam_info = json.load(f)[0]
    W, H = cam_info['width'], cam_info['height']
    fx, fy = cam_info['fx'], cam_info['fy']
    fovx = 2 * np.arctan(W / (2 * fx))
    fovy = 2 * np.arctan(H / (2 * fy))

    # 4. 加载 Seq1 的所有位姿，用于渲染 3DGS
    def load_traj_txt(file_path):
        poses = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for i in range(0, len(lines), 4):
                if i + 3 < len(lines):
                    matrix = []
                    for j in range(4):
                        matrix.append([float(x) for x in lines[i+j].strip().split()])
                    poses.append(np.array(matrix))
        return np.array(poses)

    seq1_traj_path = "/workspace_fs/guidedvd-3dgs/dataset/Replica/office_2/Sequence_1/traj_w_c.txt"
    seq1_poses = load_traj_txt(seq1_traj_path)

    metrics = {"psnr": [], "ssim": [], "lpips": []}
    lpips_model = LPIPS(net_type='alex').cuda()
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    print(f"Evaluating Top 10 frames using Sequence 1 real images as GT...")
    
    # 我们要计算的是：3DGS 模型在 Seq1 位姿下的渲染图 vs Seq1 的真实照片
    # 虽然这些位姿离机器人路径近，但它们本身在数据集里是有真值的。
    
    unique_seq1_indices = sorted(list(set([item['closest_seq1_idx'] for item in top10_data])))
    
    for seq1_idx in unique_seq1_indices:
        # A. 获取真值图
        gt_path = os.path.join(seq1_rgb_dir, f"rgb_{seq1_idx}.png")
        if not os.path.exists(gt_path):
            # 尝试其他可能的命名格式
            gt_path = os.path.join(seq1_rgb_dir, f"{seq1_idx}.png")
        
        gt_pil = Image.open(gt_path).convert("RGB")
        gt_tensor = PILtoTorch(gt_pil, (W, H)).unsqueeze(0).cuda()

        # B. 3DGS 渲染 (使用 Seq1 的位姿)
        w2c = seq1_poses[seq1_idx]
        cam = Camera(colmap_id=seq1_idx, R=w2c[:3, :3].transpose(), T=w2c[:3, 3], 
                     FoVx=fovx, FoVy=fovy, image=torch.zeros((3, H, W)), 
                     gt_alpha_mask=None, image_name=f"seq1_{seq1_idx}", uid=seq1_idx, fid=0)
        
        render_pkg = render(cam, gaussians, DummyPipe(), background)
        rendered_img = render_pkg["render"].unsqueeze(0)

        # C. 计算指标
        p = psnr(rendered_img, gt_tensor).item()
        s = ssim(rendered_img, gt_tensor).item()
        l = lpips_model(rendered_img, gt_tensor).item()
        
        metrics["psnr"].append(p)
        metrics["ssim"].append(s)
        metrics["lpips"].append(l)
        
        print(f"Seq1 Frame {seq1_idx:04d} | PSNR: {p:.2f}dB | SSIM: {s:.4f}")
        
        # 保存对比
        torchvision.utils.save_image(rendered_img.squeeze(0), os.path.join(output_dir, f"render_{seq1_idx:04d}.png"))
        gt_pil.save(os.path.join(output_dir, f"gt_{seq1_idx:04d}.png"))

    avg_psnr = np.mean(metrics["psnr"])
    print(f"\n✅ Final Average PSNR (vs Seq1 Real GT): {avg_psnr:.2f}dB")

if __name__ == "__main__":
    evaluate_top10_with_seq1_gt()
