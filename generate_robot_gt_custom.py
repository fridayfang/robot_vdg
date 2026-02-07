import os
import json
import numpy as np
import trimesh
from tqdm import tqdm
from PIL import Image

def render_scene_via_rays(mesh, camera_pose, resolution, focal):
    width, height = resolution
    scene = trimesh.Scene(mesh)
    scene.camera.resolution = resolution
    scene.camera.focal = (focal, focal)
    scene.camera_transform = camera_pose
    
    # 修正：处理 camera_rays 的不同返回格式
    res = scene.camera_rays()
    if isinstance(res, tuple):
        if len(res) == 2:
            origins, rays = res
        elif len(res) == 3:
            origins, rays, pixels = res
        else:
            origins, rays = res[0], res[1]
    else:
        # 有些版本可能直接返回 rays，origins 默认为相机位置
        rays = res
        origins = np.tile(camera_pose[:3, 3], (len(rays), 1))
    
    # 2. 射线求交
    intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
    results = intersector.intersects_id(origins, rays, multiple_hits=False, return_locations=True)
    
    if isinstance(results, tuple):
        index_tri = results[0]
        index_ray = results[1]
    else:
        index_tri = results
        index_ray = np.arange(len(index_tri))

    img_flat = np.zeros((height * width, 3), dtype=np.uint8)
    
    if len(index_tri) > 0:
        valid_mask = index_tri != -1
        index_tri = index_tri[valid_mask]
        index_ray = index_ray[valid_mask]
        
        if len(index_tri) > 0:
            faces = mesh.faces[index_tri]
            if hasattr(mesh.visual, 'vertex_colors'):
                colors = mesh.visual.vertex_colors[faces]
                avg_colors = colors.mean(axis=1)[:, :3]
                img_flat[index_ray] = avg_colors.astype(np.uint8)
            
    return img_flat.reshape((height, width, 3))

def generate_gt_final(mesh_path, json_path, output_dir, num_samples=100):
    print(f"Loading mesh from {mesh_path}...")
    mesh = trimesh.load(mesh_path)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    all_w2cs = np.array(data['w2cs_matrices'])
    
    indices = np.linspace(0, len(all_w2cs) - 1, num_samples, dtype=int)
    gt_dir = os.path.join(output_dir, "gt")
    os.makedirs(gt_dir, exist_ok=True)

    width, height = 640, 480
    hfov = np.radians(90)
    focal = width / (2 * np.tan(hfov / 2))

    print(f"Rendering {num_samples} GT images using custom Ray-Tracer...")
    for idx in tqdm(indices):
        w2c = all_w2cs[idx]
        c2w = np.linalg.inv(w2c)
        flip = np.eye(4)
        flip[1, 1], flip[2, 2] = -1, -1
        camera_pose = c2w @ flip
        
        try:
            img = render_scene_via_rays(mesh, camera_pose, (width, height), focal)
            img = np.flipud(img)
            Image.fromarray(img).save(os.path.join(gt_dir, f"{idx:04d}.png"))
        except Exception as e:
            print(f"❌ Custom render failed at {idx}: {e}")
            import traceback
            traceback.print_exc()
            break

    print(f"✅ GT images saved to {gt_dir}")

if __name__ == "__main__":
    generate_gt_final(
        "/workspace_fs/guidedvd-3dgs/dataset/Replica/office_2/mesh.ply",
        "/workspace_fs/guidedvd-3dgs/w2cs_ig.json",
        "/workspace_fs/guidedvd-3dgs/baseline_result"
    )
