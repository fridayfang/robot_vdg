import os
import json
import numpy as np
import open3d as o3d
from tqdm import tqdm
import cv2

def generate_gt_images(mesh_path, json_path, output_dir, num_samples=100):
    """
    使用 Open3D 离线渲染器从 Mesh 合成真值图像。
    """
    if not os.path.exists(mesh_path):
        print(f"Error: Mesh file not found at {mesh_path}")
        return

    # 1. 加载 Mesh
    print(f"Loading mesh from {mesh_path}...")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()

    # 2. 加载位姿并采样
    with open(json_path, 'r') as f:
        data = json.load(f)
    all_w2cs = np.array(data['w2cs_matrices'])
    
    total_poses = len(all_w2cs)
    indices = np.linspace(0, total_poses - 1, num_samples, dtype=int)
    sampled_w2cs = all_w2cs[indices]
    
    print(f"Sampled {num_samples} poses from {total_poses} total poses.")

    # 3. 初始化渲染器 (Headless)
    # 注意：在某些服务器上可能需要设置环境变量以支持 headless 渲染
    # os.environ['DISPLAY'] = ':0' 
    
    W, H = 640, 480
    render = o3d.visualization.rendering.OffscreenRenderer(W, H)
    
    # 设置场景背景为黑色
    render.scene.set_background([0, 0, 0, 1])
    
    # 设置材质
    mtl = o3d.visualization.rendering.MaterialRecord()
    mtl.base_color = [1.0, 1.0, 1.0, 1.0]
    mtl.shader = "defaultLit" # 使用默认光照以显示顶点色
    
    render.scene.add_geometry("replica_mesh", mesh, mtl)

    # 4. 渲染并保存
    gt_dir = os.path.join(output_dir, "gt")
    os.makedirs(gt_dir, exist_ok=True)
    
    print("Rendering GT images...")
    for i, idx in enumerate(tqdm(indices)):
        w2c = all_w2cs[idx]
        
        # Open3D 的外参矩阵定义可能与 COLMAP 不同
        # COLMAP: OpenCV convention (x right, y down, z forward)
        # Open3D: (x right, y up, z backward) - 需要检查
        
        # 尝试直接使用 w2c 转换
        # 在 3DGS 中，w2c 是从世界到相机的变换
        # 我们需要将其转换为 Open3D 认识的位姿
        
        # 简化处理：使用 setup_camera 的 extrinsic 参数
        # Open3D setup_camera(fov, center, eye, up)
        
        # 另一种方式：直接设置相机外参
        # render.setup_camera(intrinsic_matrix, extrinsic_matrix, width, height)
        
        # 假设 w2cs_ig.json 中的 w2c 是标准 OpenCV 格式
        # 我们需要将其转换为 Open3D 渲染器的外参
        # Open3D 的 extrinsic 是 World-to-Camera
        
        # 设置内参 (假设 90 度 HFOV)
        hfov = 90
        render.setup_camera(hfov, [0,0,0], [0,0,1], [0,-1,0]) # 默认占位
        
        # 关键：设置精确的外参
        # Open3D 的 extrinsic 矩阵是 [R | t]，World to Camera
        render.setup_camera(hfov, [0,0,0], [0,0,1], [0,-1,0]) 
        # 直接覆盖外参
        render.scene.camera.look_at([0,0,0], [0,0,1], [0,-1,0]) # 重置
        
        # 使用 look_at 逻辑转换 w2c
        c2w = np.linalg.inv(w2c)
        eye = c2w[:3, 3]
        forward = c2w[:3, 2] # Z 轴是正前方
        up = -c2w[:3, 1]     # -Y 轴是上方
        center = eye + forward
        
        render.setup_camera(hfov, center, eye, up)
        
        img = render.render_to_image()
        o3d.io.write_image(os.path.join(gt_dir, f"{idx:04d}.png"), img)

    print(f"✅ GT images saved to {gt_dir}")
    return indices

if __name__ == "__main__":
    mesh_path = "/workspace_fs/guidedvd-3dgs/dataset/Replica/office_2/mesh.ply"
    json_path = "/workspace_fs/guidedvd-3dgs/w2cs_ig.json"
    output_dir = "/workspace_fs/guidedvd-3dgs/baseline_result"
    
    generate_gt_images(mesh_path, json_path, output_dir, num_samples=100)
