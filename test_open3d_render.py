import open3d as o3d
import numpy as np
import json
import os
from tqdm import tqdm
import cv2

def test_open3d_render():
    mesh_path = "/workspace_fs/guidedvd-3dgs/dataset/Replica/office_2/mesh.ply"
    json_path = "/workspace_fs/robot_walk_2/w2cs.json"
    output_dir = "/workspace_fs/guidedvd-3dgs/gt_open3d_test"
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载 Mesh
    print("=> Loading Mesh...")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if not mesh.has_vertex_colors():
        print("⚠️ Warning: Mesh has no vertex colors!")
    mesh.compute_vertex_normals()

    # 2. 获取相机内参 (从训练配置读取)
    try:
        with open("/workspace_fs/guidedvd-3dgs/output/replica_guidedvd_office2_0203_1509/office_2/Sequence_2/cameras.json", 'r') as f:
            cam_info = json.load(f)[0]
        W, H = cam_info['width'], cam_info['height']
        fx, fy = cam_info['fx'], cam_info['fy']
        cx, cy = W/2, H/2
    except:
        W, H, fx, fy, cx, cy = 640, 480, 600, 600, 320, 240

    # 3. 初始化离线渲染器 (使用 Filament 引擎)
    print("=> Initializing OffscreenRenderer...")
    try:
        render = o3d.visualization.rendering.OffscreenRenderer(W, H)
    except Exception as e:
        print(f"❌ Failed to initialize Open3D OffscreenRenderer: {e}")
        print("Trying to set environment variables...")
        return

    # 设置材质
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit" 

    render.scene.add_geometry("mesh", mesh, material)
    render.scene.set_background([0, 0, 0, 1]) 
    
    # 设置光照
    render.scene.scene.enable_sun_light(True)
    render.scene.scene.set_sun_light([0.7, 0.7, 0.7], [1, 1, 1], 100000)

    # 4. 加载位姿并渲染前 5 帧
    with open(json_path, 'r') as f:
        data = json.load(f)
        all_w2cs = np.array(data['w2cs_matrices'] if isinstance(data, dict) else data)

    print("=> Rendering 5 frames...")
    for i in tqdm(range(5)):
        w2c = all_w2cs[i]
        c2w = np.linalg.inv(w2c)
        
        # OpenCV -> Open3D 坐标系转换
        # OpenCV: x-right, y-down, z-forward
        # Open3D: x-right, y-up, z-backward
        flip_yz = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
        c2w_o3d = c2w @ flip_yz
        
        # 设置相机
        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        render.setup_camera(intrinsic, c2w_o3d)

        img = render.render_to_image()
        o3d.io.write_image(os.path.join(output_dir, f"{i:04d}.png"), img)

    print(f"✅ Success! Images saved to {output_dir}")

if __name__ == "__main__":
    test_open3d_render()
