import numpy as np
import json
import os

def load_traj_txt(file_path):
    """加载 Replica 格式的 traj_w_c.txt (4x4 矩阵)"""
    poses = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        # 每 4 行是一个 4x4 矩阵
        for i in range(0, len(lines), 4):
            if i + 3 < len(lines):
                matrix = []
                for j in range(4):
                    matrix.append([float(x) for x in lines[i+j].strip().split()])
                poses.append(np.array(matrix))
    return np.array(poses)

def analyze_per_frame_similarity():
    # 1. 加载目标轨迹 (robot_walk_2/w2cs.json) 的前 500 帧
    target_json = "/workspace_fs/robot_walk_2/w2cs.json"
    with open(target_json, 'r') as f:
        data = json.load(f)
        target_w2cs = np.array(data['w2cs_matrices'] if isinstance(data, dict) else data)
    
    target_poses = target_w2cs[:500]
    target_centers = target_poses[:, :3, 3]

    # 2. 加载数据集序列
    seq1_path = "/workspace_fs/guidedvd-3dgs/dataset/Replica/office_2/Sequence_1/traj_w_c.txt"
    seq2_path = "/workspace_fs/guidedvd-3dgs/dataset/Replica/office_2/Sequence_2/traj_w_c.txt"
    
    seq1_poses = load_traj_txt(seq1_path)
    seq2_poses = load_traj_txt(seq2_path)
    
    seq1_centers = seq1_poses[:, :3, 3]
    seq2_centers = seq2_poses[:, :3, 3]

    # 3. 每一帧寻找 Seq1 或 Seq2 中的最近帧
    results = []
    for i in range(len(target_centers)):
        t_c = target_centers[i]
        
        # 计算到 seq1 的所有距离
        dists1 = np.linalg.norm(seq1_centers - t_c, axis=1)
        min_idx1 = np.argmin(dists1)
        min_dist1 = dists1[min_idx1]
        
        # 计算到 seq2 的所有距离
        dists2 = np.linalg.norm(seq2_centers - t_c, axis=1)
        min_idx2 = np.argmin(dists2)
        min_dist2 = dists2[min_idx2]
        
        # 综合对比，看哪个序列更近
        if min_dist1 < min_dist2:
            results.append({
                "target_frame": i,
                "best_seq": "Seq1",
                "best_idx": min_idx1,
                "dist": min_dist1
            })
        else:
            results.append({
                "target_frame": i,
                "best_seq": "Seq2",
                "best_idx": min_idx2,
                "dist": min_dist2
            })

    # 4. 统计与展示
    all_dists = [r['dist'] for r in results]
    print(f"=== Per-Frame Similarity Analysis (First 500 Frames) ===")
    print(f"Average Min Distance: {np.mean(all_dists):.4f}m")
    print(f"Max Min Distance: {np.max(all_dists):.4f}m")
    print(f"Min Min Distance: {np.min(all_dists):.4f}m")
    
    seq1_count = sum(1 for r in results if r['best_seq'] == "Seq1")
    seq2_count = sum(1 for r in results if r['best_seq'] == "Seq2")
    print(f"\nSequence Preference:")
    print(f"  Frames closer to Seq1: {seq1_count}")
    print(f"  Frames closer to Seq2: {seq2_count}")

    print("\nSample Frames (Every 100 frames):")
    for i in range(0, 500, 100):
        r = results[i]
        print(f"  Target Frame {i:03d} | Closest: {r['best_seq']} Frame {r['best_idx']:03d} | Dist: {r['dist']:.4f}m")

if __name__ == "__main__":
    analyze_per_frame_similarity()
