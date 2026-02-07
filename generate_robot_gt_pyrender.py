import os
import json
import numpy as np
import trimesh
import pyrender
from tqdm import tqdm
import imageio

# Force OSMesa for headless rendering
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

def generate_gt_pyrender(mesh_path, json_path, output_dir, num_samples=100):
    print(f"Loading mesh from {mesh_path}...")
    tm = trimesh.load(mesh_path)
    mesh = pyrender.Mesh.from_trimesh(tm)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    all_w2cs = np.array(data['w2cs_matrices'])
    
    indices = np.linspace(0, len(all_w2cs) - 1, num_samples, dtype=int)
    gt_dir = os.path.join(output_dir, "gt")
    os.makedirs(gt_dir, exist_ok=True)

    scene = pyrender.Scene(bg_color=[0, 0, 0])
    scene.add(mesh)

    # Set up camera
    # Replica cameras in 3DGS are 640x480, focal length ~320
    # HFOV = 90 deg -> fx = 320
    hfov = np.radians(90)
    camera = pyrender.PerspectiveCamera(yfov=hfov * (480/640), aspectRatio=640/480)
    camera_node = scene.add(camera)

    renderer = pyrender.OffscreenRenderer(640, 480)

    print(f"Rendering {num_samples} GT images using pyrender (OSMesa)...")
    for idx in tqdm(indices):
        w2c = all_w2cs[idx]
        c2w = np.linalg.inv(w2c)
        
        # pyrender camera convention: x right, y up, z back (same as trimesh)
        # OpenCV convention: x right, y down, z forward
        # Conversion: flip y and z
        flip = np.eye(4)
        flip[1, 1] = -1
        flip[2, 2] = -1
        
        scene.set_pose(camera_node, pose=c2w @ flip)
        
        try:
            color, _ = renderer.render(scene)
            imageio.imwrite(os.path.join(gt_dir, f"{idx:04d}.png"), color)
        except Exception as e:
            print(f"Render error at {idx}: {e}")
            break

    renderer.delete()
    print(f"✅ GT images saved to {gt_dir}")

if __name__ == "__main__":
    generate_gt_pyrender(
        "/workspace_fs/guidedvd-3dgs/dataset/Replica/office_2/mesh.ply",
        "/workspace_fs/guidedvd-3dgs/w2cs_ig.json",
        "/workspace_fs/guidedvd-3dgs/baseline_result"
    )
