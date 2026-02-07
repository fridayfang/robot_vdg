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

def evaluate_top5_seq2():
    model_path = "/workspace_fs/guidedvd-3dgs/output/replica_guidedvd_office2_0203_1509/office_2/Sequence_2"
    top5_json = "/workspace_fs/guidedvd-3dgs/top5_seq2_nearest.json"
    seq2_rgb_dir = "/workspace_fs/guidedvd-3dgs/dataset/Replica/office_2/Sequence_2/rgb"
    output_dir = "/workspace_fs/guidedvd-3dgs/eval_top5_seq2_gt"
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载 3DGS 模型
    gaussians = GaussianModel(DummyArgs())
    gaussians.load_ply(os.path.join(model_path, "point_cloud/iteration_10000/point_cloud.ply"))
    
    # 2. 加载 Top 5 信息
    with open(top5_json, 'r') as f:
        top5_data = json.load(f)

    # 3. 获取相机内参
    with open(os.path.join(model_path, "cameras.json"), 'r') as f:
        cam_info = json.load(f)[0]
    W, H = cam_info['width'], cam_info['height']
    fx, fy = cam_info['fx'], cam_info['fy']
    fovx = 2 * np.arctan(W / (2 * fx))
    fovy = 2 * np.arctan(H / (2 * fy))

    # 4. 加载 Seq2 位姿
    def load_traj_txt(file_path):
        poses = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for i in range(0, len(lines), 4):
                if i + 3 < len(lines):
                    poses.append(np.array([[float(x) for x in l.split()] for l in lines[i:i+4]]))
        return np.array(poses)

    seq2_traj_path = "/workspace_fs/guidedvd-3dgs/dataset/Replica/office_2/Sequence_2/traj_w_c.txt"
    seq2_poses = load_traj_txt(seq2_traj_path)

    metrics = {"psnr": [], "ssim": [], "lpips": []}
    lpips_model = LPIPS(net_type='alex').cuda()
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    print(f"Evaluating Top 5 frames using Sequence 2 real images as GT...")
    
    for item in top5_data:
        seq2_idx = item['seq2_idx']
        # A. 获取真值图
        gt_path = os.path.join(seq2_rgb_dir, f"rgb_{seq2_idx}.png")
        if not os.path.exists(gt_path):
            gt_path = os.path.join(seq2_rgb_dir, f"{seq2_idx}.png")
        
        gt_pil = Image.open(gt_path).convert("RGB")
        gt_tensor = PILtoTorch(gt_pil, (W, H)).unsqueeze(0).cuda()

        # B. 3DGS 渲染 (使用 Seq2 的位姿，坐标系绝对匹配)
        w2c = seq2_poses[seq2_idx]
        cam = Camera(colmap_id=seq2_idx, R=w2c[:3, :3].transpose(), T=w2c[:3, 3], 
                     FoVx=fovx, FoVy=fovy, image=torch.zeros((3, H, W)), 
                     gt_alpha_mask=None, image_name=f"seq2_{seq2_idx}", uid=seq2_idx, fid=0)
        
        render_pkg = render(cam, gaussians, DummyPipe(), background)
        rendered_img = render_pkg["render"].unsqueeze(0)

        # C. 计算指标
        p = psnr(rendered_img, gt_tensor).item()
        s = ssim(rendered_img, gt_tensor).item()
        l = lpips_model(rendered_img, gt_tensor).item()
        
        metrics["psnr"].append(p)
        metrics["ssim"].append(s)
        metrics["lpips"].append(l)
        
        print(f"Seq2 Frame {seq2_idx:04d} | PSNR: {p:.2f}dB | SSIM: {s:.4f}")
        
        # 保存对比
        torchvision.utils.save_image(rendered_img.squeeze(0), os.path.join(output_dir, f"render_{seq2_idx:04d}.png"))
        gt_pil.resize((W, H)).save(os.path.join(output_dir, f"gt_{seq2_idx:04d}.png"))

    avg_psnr = np.mean(metrics["psnr"])
    print(f"\n✅ Final Average PSNR (vs Seq2 Real GT): {avg_psnr:.2f}dB")

    # 保存结果供可视化
    save_data = {"per_frame_psnr": metrics["psnr"]}
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(save_data, f)

if __name__ == "__main__":
    evaluate_top5_seq2()
