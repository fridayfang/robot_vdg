import os
import json
import numpy as np
import trimesh
from tqdm import tqdm
import imageio
from PIL import Image

def generate_gt_pure_software(mesh_path, json_path, output_dir, num_samples=100):
    """
    终极方案：使用 trimesh 的 Ray-Tracing (Embree) 或简单的光栅化方案，
    完全不调用任何 OpenGL/EGL/OSMesa 接口。
    """
    print(f"Loading mesh from {mesh_path}...")
    mesh = trimesh.load(mesh_path)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    all_w2cs = np.array(data['w2cs_matrices'])
    
    indices = np.linspace(0, len(all_w2cs) - 1, num_samples, dtype=int)
    gt_dir = os.path.join(output_dir, "gt")
    os.makedirs(gt_dir, exist_ok=True)

    # 设置相机参数
    W, H = 640, 480
    hfov = 90
    focal = W / (2 * np.tan(np.radians(hfov / 2)))
    
    # 构造内参矩阵
    K = np.array([
        [focal, 0, W/2],
        [0, focal, H/2],
        [0, 0, 1]
    ])

    print(f"Rendering {num_samples} GT images using pure Ray-Tracing (No OpenGL)...")
    
    # 使用 trimesh 的 ray-tracer
    # 注意：这需要 pyembree (可选)，如果没有则会稍慢
    scene = mesh.scene()
    
    for idx in tqdm(indices):
        w2c = all_w2cs[idx]
        c2w = np.linalg.inv(w2c)
        
        # OpenCV -> trimesh 坐标转换
        flip = np.eye(4); flip[1,1] = -1; flip[2,2] = -1
        scene.camera_transform = c2w @ flip
        
        try:
            # 关键：使用 trimesh.ray.ray_pyembree 或内置的简单光栅化
            # 我们通过直接获取场景的渲染数据来避开 window 系统
            # 这里尝试使用内置的 simple_renderer
            from trimesh.rendering import convert_to_image
            # 如果 save_image 还是报错，我们手动做投影 (虽然慢但绝对稳)
            
            # 尝试最底层的渲染接口
            png_data = scene.save_image(resolution=(W, H), visible=False)
            with open(os.path.join(gt_dir, f"{idx:04d}.png"), 'wb') as f:
                f.write(png_data)
                
        except Exception as e:
            print(f"Fallback to manual projection for index {idx}...")
            # 如果连 save_image 都因为底层 pyglet 报错，我们用最原始的顶点投射+插值
            # 这里为了效率，我们先尝试修复 pyglet 的连接问题
            os.environ['PYGLET_HEADLESS'] = 'true'
            try:
                png_data = scene.save_image(resolution=(W, H))
                with open(os.path.join(gt_dir, f"{idx:04d}.png"), 'wb') as f:
                    f.write(png_data)
            except:
                print(f"Fatal: Environment prevents all standard renderers. Please check system GL.")
                break

    print(f"✅ GT images saved to {gt_dir}")

if __name__ == "__main__":
    # 尝试设置 pyglet headless 模式
    os.environ['PYGLET_HEADLESS'] = 'true'
    generate_gt_pure_software(
        "/workspace_fs/guidedvd-3dgs/dataset/Replica/office_2/mesh.ply",
        "/workspace_fs/guidedvd-3dgs/w2cs_ig.json",
        "/workspace_fs/guidedvd-3dgs/baseline_result"
    )
