import os
import json
import numpy as np
import trimesh
from tqdm import tqdm
import imageio
from PIL import Image, ImageDraw

def generate_gt_projection(mesh_path, json_path, output_dir, num_samples=100):
    """
    终极方案：手动顶点投影。
    将 Mesh 的顶点投影到像素平面，并使用顶点色填充。
    虽然没有遮挡剔除和光照，但对于定性分析和简单的 PSNR 计算（在稀疏点云背景下）是唯一的出路。
    """
    print(f"Loading mesh from {mesh_path}...")
    mesh = trimesh.load(mesh_path)
    vertices = mesh.vertices
    colors = mesh.visual.vertex_colors[:, :3] # [N, 3]
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    all_w2cs = np.array(data['w2cs_matrices'])
    
    indices = np.linspace(0, len(all_w2cs) - 1, num_samples, dtype=int)
    gt_dir = os.path.join(output_dir, "gt")
    os.makedirs(gt_dir, exist_ok=True)

    W, H = 640, 480
    hfov = 90
    focal = W / (2 * np.tan(np.radians(hfov / 2)))
    K = np.array([
        [focal, 0, W/2],
        [0, focal, H/2],
        [0, 0, 1]
    ])

    print(f"Projecting {num_samples} views manually (No OpenGL dependency)...")
    for idx in tqdm(indices):
        w2c = all_w2cs[idx]
        
        # 1. 变换顶点到相机坐标系
        # vertices: [N, 3] -> [N, 4]
        v_homo = np.hstack([vertices, np.ones((vertices.shape[0], 1))])
        v_cam = (w2c @ v_homo.T).T # [N, 4]
        
        # 2. 投影到像素平面
        # 仅保留 z > 0 的点 (在相机前方)
        mask = v_cam[:, 2] > 0.1
        v_cam_fwd = v_cam[mask]
        v_colors = colors[mask]
        
        # 归一化坐标
        pts_2d = (K @ v_cam_fwd[:, :3].T).T
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        
        # 3. 绘制图像
        img = np.zeros((H, W, 3), dtype=np.uint8)
        
        # 仅保留在图像范围内的点
        valid_mask = (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < W) & \
                     (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < H)
        
        final_pts = pts_2d[valid_mask].astype(int)
        final_colors = v_colors[valid_mask]
        
        # 简单的散点绘制 (这里不处理遮挡，后画的点会覆盖前面的)
        # 为了效果好一点，我们按深度排序
        depths = v_cam_fwd[valid_mask, 2]
        sort_idx = np.argsort(-depths) # 从远到近画
        
        img[final_pts[sort_idx, 1], final_pts[sort_idx, 0]] = final_colors[sort_idx]
        
        # 膨胀一下点，让图看起来不那么稀疏
        kernel = np.ones((3,3), np.uint8)
        import cv2
        img = cv2.dilate(img, kernel, iterations=1)

        imageio.imwrite(os.path.join(gt_dir, f"{idx:04d}.png"), img)

    print(f"✅ Manual projection saved to {gt_dir}")

if __name__ == "__main__":
    generate_gt_projection(
        "/workspace_fs/guidedvd-3dgs/dataset/Replica/office_2/mesh.ply",
        "/workspace_fs/guidedvd-3dgs/w2cs_ig.json",
        "/workspace_fs/guidedvd-3dgs/baseline_result"
    )
