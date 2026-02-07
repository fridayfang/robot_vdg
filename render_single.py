import torch
import os
import json
import numpy as np
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
import torchvision

class DummyArgs:
    def __init__(self): self.sh_degree = 3
class DummyPipe:
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.use_confidence = False

def render_single_frame(model_path, iteration, json_path, frame_idx=0, output_path="first_frame_render.png"):
    # 1. 加载模型
    gaussians = GaussianModel(DummyArgs())
    ply_path = os.path.join(model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
    gaussians.load_ply(ply_path)
    
    # 2. 加载位姿
    with open(json_path, 'r') as f:
        data = json.load(f)
        # 兼容两种格式：直接是矩阵列表，或者是在 'w2cs_matrices' 键下
        if isinstance(data, dict) and 'w2cs_matrices' in data:
            all_w2cs = np.array(data['w2cs_matrices'])
        else:
            all_w2cs = np.array(data)
    
    w2c = all_w2cs[frame_idx]
    print(f"Rendering frame {frame_idx} from {json_path}")

    # 3. 获取相机内参 (从训练时的 cameras.json 读取)
    with open(os.path.join(model_path, "cameras.json"), 'r') as f:
        cam_info = json.load(f)[0]
    W, H = cam_info['width'], cam_info['height']
    # 计算 FoV
    fovx = 2 * np.arctan(W / (2 * cam_info['fx']))
    fovy = 2 * np.arctan(H / (2 * cam_info['fy']))

    # 4. 渲染
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    # 注意：Camera 类构造函数中 R 是旋转矩阵，T 是平移向量
    # w2c[:3, :3] 是 R_w2c，其转置是 R_c2w (即 Camera 类需要的 R)
    cam = Camera(colmap_id=frame_idx, R=w2c[:3, :3].transpose(), T=w2c[:3, 3], 
                 FoVx=fovx, FoVy=fovy, image=torch.zeros((3, H, W)), 
                 gt_alpha_mask=None, image_name=f"render_{frame_idx}", 
                 uid=frame_idx, fid=0)
    
    render_pkg = render(cam, gaussians, DummyPipe(), background)
    rendered_img = render_pkg["render"]
    
    # 5. 保存
    torchvision.utils.save_image(rendered_img, output_path)
    print(f"✅ Rendered image saved to {output_path}")

if __name__ == "__main__":
    render_single_frame(
        model_path="/workspace_fs/guidedvd-3dgs/output/replica_guidedvd_office2_0203_1509/office_2/Sequence_2",
        iteration=10000,
        json_path="/workspace_fs/robot_walk_2/w2cs.json",
        frame_idx=0,
        output_path="/workspace_fs/robot_walk_2_first_frame.png"
    )
