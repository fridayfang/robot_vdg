import os
import sys
import numpy as np
import json
import trimesh
from tqdm import tqdm
import imageio

# --- 核心环境修复代码 ---
def fix_opengl_environment():
    # 1. 强制指定 OSMesa 平台
    os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
    
    # 2. 尝试寻找 Conda 环境中的 OSMesa 库
    conda_prefix = os.environ.get('CONDA_PREFIX', '/opt/conda')
    lib_path = os.path.join(conda_prefix, 'lib')
    osmesa_lib = os.path.join(lib_path, 'libOSMesa.so')
    
    if os.path.exists(osmesa_lib):
        # 强制将 Conda 的 lib 路径加入搜索首位
        os.environ['LD_LIBRARY_PATH'] = lib_path + ':' + os.environ.get('LD_LIBRARY_PATH', '')
        print(f"✅ Found OSMesa at {osmesa_lib}, added to LD_LIBRARY_PATH")
    else:
        print("⚠️ OSMesa library not found in Conda, trying system default...")

# 在导入 pyrender 之前执行修复
fix_opengl_environment()

try:
    import pyrender
    print("✅ pyrender imported successfully")
except Exception as e:
    print(f"❌ pyrender import failed: {e}")
    # 如果 pyrender 失败，我们将回退到 trimesh 的纯软件渲染方案
    pass

def generate_gt_pyrender(mesh_path, json_path, output_dir, num_samples=100):
    print(f"Loading mesh from {mesh_path}...")
    tm = trimesh.load(mesh_path)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    all_w2cs = np.array(data['w2cs_matrices'])
    
    indices = np.linspace(0, len(all_w2cs) - 1, num_samples, dtype=int)
    gt_dir = os.path.join(output_dir, "gt")
    os.makedirs(gt_dir, exist_ok=True)

    try:
        # 尝试使用 pyrender (硬件/软件加速)
        mesh = pyrender.Mesh.from_trimesh(tm)
        scene = pyrender.Scene(bg_color=[0, 0, 0])
        scene.add(mesh)
        hfov = np.radians(90)
        camera = pyrender.PerspectiveCamera(yfov=hfov * (480/640), aspectRatio=640/480)
        camera_node = scene.add(camera)
        renderer = pyrender.OffscreenRenderer(640, 480)

        print(f"Rendering {num_samples} GT images using pyrender...")
        for idx in tqdm(indices):
            w2c = all_w2cs[idx]
            c2w = np.linalg.inv(w2c)
            flip = np.eye(4); flip[1,1] = -1; flip[2,2] = -1
            scene.set_pose(camera_node, pose=c2w @ flip)
            color, _ = renderer.render(scene)
            imageio.imwrite(os.path.join(gt_dir, f"{idx:04d}.png"), color)
        renderer.delete()
        
    except Exception as e:
        print(f"⚠️ pyrender failed ({e}), falling back to trimesh software rasterizer...")
        # 终极回退：trimesh 纯软件渲染 (不需要任何 OpenGL)
        scene = tm.scene()
        hfov = 90
        vfov = 2 * np.degrees(np.arctan(np.tan(np.radians(hfov/2)) * (480/640)))
        scene.camera.fov = [hfov, vfov]
        
        for idx in tqdm(indices):
            w2c = all_w2cs[idx]
            c2w = np.linalg.inv(w2c)
            flip = np.eye(4); flip[1,1] = -1; flip[2,2] = -1
            scene.camera_transform = c2w @ flip
            # trimesh.scene.save_image 在没有渲染器时会尝试使用简单的内置光栅化
            try:
                png_data = scene.save_image(resolution=(640, 480))
                with open(os.path.join(gt_dir, f"{idx:04d}.png"), 'wb') as f:
                    f.write(png_data)
            except Exception as e2:
                print(f"❌ Software render also failed at {idx}: {e2}")
                break

    print(f"✅ GT images saved to {gt_dir}")

if __name__ == "__main__":
    generate_gt_pyrender(
        "/workspace_fs/guidedvd-3dgs/dataset/Replica/office_2/mesh.ply",
        "/workspace_fs/guidedvd-3dgs/w2cs_ig.json",
        "/workspace_fs/guidedvd-3dgs/baseline_result"
    )
