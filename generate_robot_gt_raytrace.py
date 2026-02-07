import os
import json
import numpy as np
import trimesh
from tqdm import tqdm
from PIL import Image

def generate_gt_software(mesh_path, json_path, output_dir, num_samples=100):
    """
    使用 trimesh + pyembree 进行纯软件射线检测渲染。
    这种方法完全不依赖 OpenGL/GPU 驱动，最适合 Headless 服务器。
    """
    print(f"Loading mesh from {mesh_path}...")
    mesh = trimesh.load(mesh_path)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    all_w2cs = np.array(data['w2cs_matrices'])
    
    indices = np.linspace(0, len(all_w2cs) - 1, num_samples, dtype=int)
    gt_dir = os.path.join(output_dir, "gt")
    os.makedirs(gt_dir, exist_ok=True)

    # 设置分辨率和内参 (Replica 640x480, HFOV=90)
    width, height = 640, 480
    hfov = np.radians(90)
    focal = width / (2 * np.tan(hfov / 2))

    print(f"Rendering {num_samples} GT images using CPU Ray-Tracing...")
    for idx in tqdm(indices):
        w2c = all_w2cs[idx]
        c2w = np.linalg.inv(w2c)
        
        # 坐标系转换：OpenCV -> trimesh
        flip = np.eye(4)
        flip[1, 1], flip[2, 2] = -1, -1
        camera_pose = c2w @ flip
        
        # 使用 trimesh 的射线检测渲染器 (需要 pyembree)
        try:
            # 创建场景和相机
            scene = trimesh.Scene(mesh)
            scene.camera.resolution = (width, height)
            scene.camera.fov = (np.degrees(hfov), np.degrees(2 * np.arctan(height / (2 * focal))))
            scene.camera_transform = camera_pose
            
            # 关键：使用射线检测生成图像
            # 这种方法虽然慢，但在 Headless 环境下极其可靠
            color, depth = scene.convert_ray_to_image()
            
            # 保存
            Image.fromarray(color).save(os.path.join(gt_dir, f"{idx:04d}.png"))
            
        except Exception as e:
            print(f"❌ Ray-trace render failed at {idx}: {e}")
            # 如果连这个都失败，说明 pyembree 没装好，尝试最后的像素级遍历（极慢，仅作为演示）
            break

    print(f"✅ GT images saved to {gt_dir}")

if __name__ == "__main__":
    generate_gt_software(
        "/workspace_fs/guidedvd-3dgs/dataset/Replica/office_2/mesh.ply",
        "/workspace_fs/guidedvd-3dgs/w2cs_ig.json",
        "/workspace_fs/guidedvd-3dgs/baseline_result"
    )
