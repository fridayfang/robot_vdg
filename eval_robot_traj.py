import torch
import os
import json
import numpy as np
from tqdm import tqdm
from scene import Scene
from gaussian_renderer import render
from utils.general_utils import PILtoTorch
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
from scene.cameras import Camera
import torchvision

def evaluate_robot_trajectory(model_path, iteration, json_path, output_dir):
    # 1. Load the model and scene
    # We need the original scene info to get intrinsics
    # But since we are rendering from arbitrary poses, we can just use the intrinsics from the first training camera
    
    # Load cameras.json to get intrinsics
    cameras_json_path = os.path.join(model_path, "cameras.json")
    with open(cameras_json_path, 'r') as f:
        cam_info = json.load(f)[0]
    
    W, H = cam_info['width'], cam_info['height']
    fx, fy = cam_info['fx'], cam_info['fy']
    fovx = 2 * np.arctan(W / (2 * fx))
    fovy = 2 * np.arctan(H / (2 * fy))

    # Load the Gaussian model
    from scene.gaussian_model import GaussianModel
    class DummyArgs:
        def __init__(self):
            self.sh_degree = 3
    gaussians = GaussianModel(DummyArgs())
    ply_path = os.path.join(model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
    gaussians.load_ply(ply_path)
    
    # Load the robot poses
    with open(json_path, 'r') as f:
        data = json.load(f)
    w2cs = np.array(data['w2cs_matrices'])
    
    os.makedirs(output_dir, exist_ok=True)
    render_dir = os.path.join(output_dir, "renders")
    os.makedirs(render_dir, exist_ok=True)
    
    print(f"Rendering {len(w2cs)} robot views...")
    
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    
    class DummyPipe:
        def __init__(self):
            self.convert_SHs_python = False
            self.compute_cov3D_python = False
            self.debug = False
            self.use_confidence = False
    pipe = DummyPipe()
    
    results = []
    
    for i, w2c in enumerate(tqdm(w2cs)):
        # Convert W2C to R, T for 3DGS camera
        # w2c is 4x4
        R = w2c[:3, :3].transpose() # 3DGS expects R transposed
        T = w2c[:3, 3]
        
        # Create a camera object
        cam = Camera(colmap_id=i, R=R, T=T, FoVx=fovx, FoVy=fovy, 
                     image=torch.zeros((3, H, W)), # Dummy image
                     gt_alpha_mask=None, image_name=f"robot_{i}", 
                     uid=i, fid=0, data_device="cuda")
        
        # Render
        render_pkg = render(cam, gaussians, pipe, background)
        image = render_pkg["render"] # [3, H, W]
        
        # Save image
        torchvision.utils.save_image(image, os.path.join(render_dir, f"{i:04d}.png"))
        
        # If we had ground truth, we would calculate metrics here
        # results.append({"id": i, "psnr": ..., "ssim": ..., "lpips": ...})

    print(f"✅ Finished rendering. Images saved to {render_dir}")

if __name__ == "__main__":
    model_path = "/workspace_fs/guidedvd-3dgs/output/replica_guidedvd_office2_0203_1509/office_2/Sequence_2"
    iteration = 10000
    json_path = "/workspace_fs/guidedvd-3dgs/w2cs_ig.json"
    output_dir = "/workspace_fs/guidedvd-3dgs/baseline_result"
    
    evaluate_robot_trajectory(model_path, iteration, json_path, output_dir)
