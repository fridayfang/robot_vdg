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

def evaluate_custom(model_path, iteration, w2cs_path, gt_images_path, total_frames=200, num_samples=20):
    from arguments import ModelParams
    from argparse import Namespace
    from scene import Scene, GaussianModel
    from scene.cameras import Camera
    from utils.general_utils import PILtoTorch
    
    # 1. Load training config
    checkpoint_path = os.path.join(model_path, "cfg_args")
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            train_args = eval(f.read())
    else:
        raise FileNotFoundError(f"cfg_args not found in {model_path}")

    # 2. Initialize Gaussians and Scene
    gaussians = GaussianModel(train_args)
    scene = Scene(train_args, gaussians, load_iteration=iteration, shuffle=False)
    
    # 3. Load Walk Trajectory Poses
    with open(w2cs_path, 'r') as f:
        data = json.load(f)
        all_w2cs = np.array(data['w2cs_matrices'])
        
    indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
    
    # 4. Camera Intrinsic Settings (matched to 518x518 GT)
    width, height = 518, 518
    # Discovery: The 518x518 images are likely a resize of the original 640x480 images.
    # Original focal 320 for width 640 -> New focal = 320 * (518 / 640) = 259.0
    focal = 259.0 
    fovx = focal2fov(focal, width)
    fovy = focal2fov(focal, height)
    
    psnrs, ssims, lpipss = [], [], []
    output_dir = os.path.join(model_path, f"final_strict_eval_top{total_frames}_s{num_samples}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Starting STRICT evaluation for {num_samples} samples...")
    
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    pipeline_params = Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, use_confidence=False, use_color=True)

    for i, frame_idx in enumerate(tqdm(indices)):
        w2c = all_w2cs[frame_idx]
        
        # Consistent with render_single.py and training:
        # Pass R = w2c[:3, :3].T to Camera class
        R = w2c[:3, :3].transpose()
        T = w2c[:3, 3]
        
        # Load GT
        gt_path = os.path.join(gt_images_path, f"{frame_idx:06d}.png")
        gt_image = Image.open(gt_path).convert("RGB")
        gt_tensor = tf.to_tensor(gt_image).cuda()
        
        # Create standard Camera object (reuse project's class and normalization)
        # We pass the trans/scale from the first training camera to ensure alignment
        train_cam_sample = scene.getTrainCameras()[0]
        
        cam = Camera(
            colmap_id=frame_idx, 
            R=R, T=T, 
            FoVx=fovx, FoVy=fovy, 
            image=torch.zeros((3, height, width)), # Dummy image for init
            gt_alpha_mask=None,
            image_name=f"{frame_idx:06d}",
            uid=i, fid=i,
            trans=train_cam_sample.trans,
            scale=train_cam_sample.scale,
            data_device="cuda"
        )
        
        # Render using the standard method
        render_pkg = render(cam, gaussians, pipeline_params, bg_color)
        image = torch.clamp(render_pkg["render"], 0.0, 1.0)
        
        # Save comparison
        torchvision_save_path = os.path.join(output_dir, f"sample_{i:02d}_frame_{frame_idx:06d}.png")
        combined = torch.cat([image, gt_tensor], dim=2)
        import torchvision
        torchvision.utils.save_image(combined, torchvision_save_path)

        # Metrics
        _psnr = psnr(image, gt_tensor).mean().item()
        _ssim = ssim(image.unsqueeze(0), gt_tensor.unsqueeze(0)).item()
        _lpips = lpips(image.unsqueeze(0), gt_tensor.unsqueeze(0), net_type='vgg').item()
        
        psnrs.append(_psnr)
        ssims.append(_ssim)
        lpipss.append(_lpips)
        
    avg_psnr = np.mean(psnrs)
    avg_ssim = np.mean(ssims)
    avg_lpips = np.mean(lpipss)
    
    print("\n" + "="*30)
    print(f"Evaluation Results ({num_samples} samples from first {total_frames} frames):")
    print(f"Average PSNR: {avg_psnr:.4f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average LPIPS: {avg_lpips:.4f}")
    print("="*30)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--iteration", type=int, default=10000)
    parser.add_argument("--w2cs", type=str, required=True)
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--total_frames", type=int, default=200)
    parser.add_argument("--num_samples", type=int, default=20)
    args = parser.parse_args()
    
    evaluate_custom(args.model_path, args.iteration, args.w2cs, args.gt_dir, args.total_frames, args.num_samples)
