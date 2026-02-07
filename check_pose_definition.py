import numpy as np
import json
import os

def load_traj_txt_correct(file_path):
    """加载 Replica 格式的 traj_w_c.txt (每行 16 个数，展平的 4x4 矩阵)"""
    poses = []
    with open(file_path, 'r') as f:
        for line in f:
            nums = [float(x) for x in line.strip().split()]
            if len(nums) == 16:
                poses.append(np.array(nums).reshape(4, 4))
    return np.array(poses)

def check_poses():
    model_path = "/workspace_fs/guidedvd-3dgs/output/replica_guidedvd_office2_0203_1509/office_2/Sequence_2"
    traj_path = "/workspace_fs/guidedvd-3dgs/dataset/Replica/office_2/Sequence_2/traj_w_c.txt"
    
    # 1. 从 cameras.json 读取训练位姿 (C2W)
    with open(os.path.join(model_path, "cameras.json"), 'r') as f:
        cams_info = json.load(f)
    
    # 找到 rgb_194 对应的项 (这是 cameras.json 中的第一项)
    cam_info = cams_info[0]
    img_name = cam_info['img_name']
    # 假设 img_name 是 'rgb_194'，那么它对应原始序列的第 194 帧
    try:
        orig_idx = int(img_name.replace('rgb_', '').replace('.png', ''))
    except:
        orig_idx = 0

    print(f"--- From cameras.json (Name: {img_name}, Extracted Index: {orig_idx}) ---")
    pos = np.array(cam_info['position'])
    rot = np.array(cam_info['rotation'])
    c2w_json = np.eye(4)
    c2w_json[:3, :3] = rot
    c2w_json[:3, 3] = pos
    print("C2W Matrix:\n", c2w_json)

    # 2. 从 traj_w_c.txt 读取对应的原始帧
    traj_poses = load_traj_txt_correct(traj_path)
    if orig_idx < len(traj_poses):
        pose_orig_traj = traj_poses[orig_idx]
        print(f"\n--- From traj_w_c.txt (Frame {orig_idx}) ---")
        print("Matrix in file:\n", pose_orig_traj)
        
        # 3. 尝试各种转换，看哪个能对上
        print("\n--- Comparison Analysis ---")
        # 检查是否是逆关系
        if np.allclose(c2w_json, pose_orig_traj, atol=1e-2):
            print("MATCH: traj_w_c.txt is C2W (same as cameras.json)")
        elif np.allclose(np.linalg.inv(c2w_json), pose_orig_traj, atol=1e-2):
            print("MATCH: traj_w_c.txt is W2C (inverse of cameras.json)")
        else:
            # 检查轴翻转 (OpenCV vs OpenGL)
            flip_yz = np.diag([1, -1, -1, 1])
            c2w_flipped = c2w_json @ flip_yz
            if np.allclose(c2w_flipped, pose_orig_traj, atol=1e-2):
                print("MATCH: traj_w_c.txt is C2W with YZ flip (OpenGL format)")
            elif np.allclose(np.linalg.inv(c2w_flipped), pose_orig_traj, atol=1e-2):
                print("MATCH: traj_w_c.txt is W2C with YZ flip")
            else:
                print("NO DIRECT MATCH. Checking translation only...")
                print("JSON Position:", pos)
                print("Traj Position (col 4):", pose_orig_traj[:3, 3])
                # 检查平移是否是 W2C 的平移
                t_w2c_from_json = np.linalg.inv(c2w_json)[:3, 3]
                print("W2C Translation from JSON:", t_w2c_from_json)
    else:
        print(f"Error: Index {orig_idx} out of range for traj_w_c.txt (length {len(traj_poses)})")

if __name__ == "__main__":
    check_poses()
