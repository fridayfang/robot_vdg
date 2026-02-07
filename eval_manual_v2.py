import torch
import torchvision
import os
import json
import numpy as np
from tqdm import tqdm
import trimesh
from PIL import Image
import imageio
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

def render_mesh_manual_fast(mesh, w2c, K, H, W):
    """
    纯 CPU 顶点投影 + 深度缓冲 (Z-Buffer)
    虽然不如 OpenGL 平滑，但比之前的点投影更密，且完全无驱动依赖。
    """
    vertices = mesh.vertices
    faces = mesh.faces
    # 如果 mesh 有顶点颜色，使用它；否则默认灰色
    if hasattr(mesh.visual, 'vertex_colors'):
        colors = mesh.visual.vertex_colors[:, :3] / 255.0
    else:
        colors = np.ones_like(vertices) * 0.5

    R = w2c[:3, :3]
    t = w2c[:3, 3]
    
    # 转换到相机空间
    v_cam = (R @ vertices.T).T + t
    
    # 投影到像素空间
    v_img_h = (K @ v_cam.T).T
    depths = v_img_h[:, 2]
    
    # 过滤掉相机背后的点
    mask = depths > 0.1
    v_img = v_img_h[mask, :2] / depths[mask, None]
    c_img = colors[mask]
    d_img = depths[mask]
    
    # 过滤掉画布外的点
    x, y = v_img[:, 0], v_img[:, 1]
    valid = (x >= 0) & (x < W-1) & (y >= 0) & (y < H-1)
    
    final_x = x[valid].astype(int)
    final_y = y[valid].astype(int)
    final_c = c_img[valid]
    final_d = d_img[valid]
    
    img = np.zeros((H, W, 3), dtype=np.uint8)
    z_buf = np.full((H, W), np.inf, dtype=np.float32)
    
    # 按深度排序，从远到近绘制（简单的覆盖逻辑）
    indices = np.argsort(final_d)[::-1]
    for i in indices:
        px, py = final_x[i], final_y[i]
        if final_d[i] < z_buf[py, px]:
            img[py, px] = (final_c[i] * 255).astype(np.uint8)
            z_buf[py, px] = final_d[i]
            # 简单的膨胀，让点云看起来更像面片
            img[py+1, px] = img[py, px]
            img[py, px+1] = img[py, px]
            
    return img

def evaluate_with_manual_mesh_gt(model_path, iteration, json_path, mesh_path, output_dir, num_samples=5):
    gaussians = GaussianModel(DummyArgs())
    gaussians.load_ply(os.path.join(model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply"))
    
    with open(json_path, 'r') as f:
        data = json.load(f)
        all_w2cs = np.array(data['w2cs_matrices'] if isinstance(data, dict) else data)
    sampled_w2cs = all_w2cs[:num_samples]

    with open(os.path.join(model_path, "cameras.json"), 'r') as f:
        cam_info = json.load(f)[0]
    W, H = cam_info['width'], cam_info['height']
    fx, fy = cam_info['fx'], cam_info['fy']
    K = np.array([[fx, 0, W/2], [0, fy, H/2], [0, 0, 1]])

    print(f"Loading mesh: {mesh_path}")
    mesh = trimesh.load(mesh_path)
    
    gt_dir = os.path.join(output_dir, "gt_manual")
    render_dir = os.path.join(output_dir, "renders")
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(render_dir, exist_ok=True)

    metrics = {"psnr": [], "ssim": [], "lpips": []}
    lpips_model = LPIPS(net_type='alex').cuda()
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    print(f"Generating Manual GT and Evaluating {num_samples} frames...")
    for i in range(num_samples):
        w2c = sampled_w2cs[i]
        
        # 1. 手动投影生成 GT
        gt_img_np = render_mesh_manual_fast(mesh, w2c, K, H, W)
        imageio.imwrite(os.path.join(gt_dir, f"{i:04d}.png"), gt_img_np)
        gt_tensor = PILtoTorch(Image.fromarray(gt_img_np), (W, H)).unsqueeze(0).cuda()

        # 2. 3DGS 渲染
        fovx = 2 * np.arctan(W / (2 * fx))
        fovy = 2 * np.arctan(H / (2 * fy))
        cam = Camera(colmap_id=i, R=w2c[:3, :3].transpose(), T=w2c[:3, 3], 
                     FoVx=fovx, FoVy=fovy, image=torch.zeros((3, H, W)), 
                     gt_alpha_mask=None, image_name=f"eval_{i}", uid=i, fid=0)
        render_pkg = render(cam, gaussians, DummyPipe(), background)
        rendered_img = render_pkg["render"].unsqueeze(0)
        torchvision.utils.save_image(rendered_img.squeeze(0), os.path.join(render_dir, f"{i:04d}.png"))

        # 3. 计算指标
        p = psnr(rendered_img, gt_tensor).item()
        s = ssim(rendered_img, gt_tensor).item()
        l = lpips_model(rendered_img, gt_tensor).item()
        
        metrics["psnr"].append(p)
        metrics["ssim"].append(s)
        metrics["lpips"].append(l)
        print(f"Frame {i:04d} | PSNR: {p:.2f}dB | SSIM: {s:.4f}")

    avg_psnr = np.mean(metrics["psnr"])
    print(f"\n✅ Final Average PSNR: {avg_psnr:.2f}dB")

if __name__ == "__main__":
    evaluate_with_manual_mesh_gt(
        model_path="/workspace_fs/guidedvd-3dgs/output/replica_guidedvd_office2_0203_1509/office_2/Sequence_2",
        iteration=10000,
        json_path="/workspace_fs/robot_walk_2/w2cs.json",
        mesh_path="/workspace_fs/guidedvd-3dgs/dataset/Replica/office_2/mesh.ply",
        output_dir="/workspace_fs/guidedvd-3dgs/eval_manual_v2",
        num_samples=5
    )
