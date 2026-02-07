import os
import json
import numpy as np
import trimesh
from tqdm import tqdm
import imageio

def generate_gt_v2(mesh_path, json_path, output_dir, num_samples=100):
    print(f"Loading mesh from {mesh_path}...")
    # 使用 trimesh 加载
    mesh = trimesh.load(mesh_path)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    all_w2cs = np.array(data['w2cs_matrices'])
    
    total_poses = len(all_w2cs)
    indices = np.linspace(0, total_poses - 1, num_samples, dtype=int)
    gt_dir = os.path.join(output_dir, "gt")
    os.makedirs(gt_dir, exist_ok=True)

    # 创建场景
    scene = trimesh.Scene(mesh)
    
    # 设置相机参数 (640x480, 90 deg HFOV)
    # trimesh.scene.Camera(fov=[hfov, vfov])
    hfov = 90
    vfov = 2 * np.degrees(np.arctan(np.tan(np.radians(hfov/2)) * (480/640)))
    scene.camera.fov = [hfov, vfov]

    print(f"Rendering {num_samples} GT images using software renderer...")
    for idx in tqdm(indices):
        w2c = all_w2cs[idx]
        c2w = np.linalg.inv(w2c)
        
        # 坐标系转换：OpenCV (x-right, y-down, z-forward) -> trimesh (x-right, y-up, z-back)
        # 1. 绕 X 轴旋转 180 度 (y-down -> y-up, z-forward -> z-back)
        flip_yz = np.eye(4)
        flip_yz[1, 1] = -1
        flip_yz[2, 2] = -1
        
        scene.camera_transform = c2w @ flip_yz
        
        try:
            # 使用 trimesh 的 save_image，它会自动处理 headless 渲染
            # 如果没有安装 pyrender，它会回退到简单的渲染方式
            png_data = scene.save_image(resolution=(640, 480))
            with open(os.path.join(gt_dir, f"{idx:04d}.png"), 'wb') as f:
                f.write(png_data)
        except Exception as e:
            # 如果 save_image 失败，尝试用 pyrender 强制 headless
            try:
                import pyrender
                os.environ['PYOPENGL_PLATFORM'] = 'egl' # 尝试 EGL
                r = pyrender.OffscreenRenderer(640, 480)
                # ... 这里的备选方案较复杂，先看 trimesh 能否成功
                print(f"Render error at {idx}: {e}")
                break
            except:
                print(f"Fatal render error at {idx}: {e}")
                break

    print(f"✅ GT images saved to {gt_dir}")
    return indices

if __name__ == "__main__":
    generate_gt_v2(
        "/workspace_fs/guidedvd-3dgs/dataset/Replica/office_2/mesh.ply",
        "/workspace_fs/guidedvd-3dgs/w2cs_ig.json",
        "/workspace_fs/guidedvd-3dgs/baseline_result"
    )
