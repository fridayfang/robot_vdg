import numpy as np
import json
import os
from tqdm import tqdm

def load_traj_txt(file_path):
    """加载 Replica 格式的 traj_w_c.txt (4x4 矩阵)"""
    poses = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 4):
            if i + 3 < len(lines):
                matrix = []
                for j in range(4):
                    matrix.append([float(x) for x in lines[i+j].strip().split()])
                poses.append(np.array(matrix))
    return np.array(poses)

def analyze_seq1_similarity():
    # 1. 加载目标轨迹 (robot_walk_2/w2cs.json) 的前 500 帧
    target_json = "/workspace_fs/robot_walk_2/w2cs.json"
    with open(target_json, 'r') as f:
        data = json.load(f)
        target_w2cs = np.array(data['w2cs_matrices'] if isinstance(data, dict) else data)
    
    target_poses = target_w2cs[:500]
    target_centers = target_poses[:, :3, 3]

    # 2. 加载数据集 Sequence 1
    seq1_path = "/workspace_fs/guidedvd-3dgs/dataset/Replica/office_2/Sequence_1/traj_w_c.txt"
    seq1_poses = load_traj_txt(seq1_path)
    seq1_centers = seq1_poses[:, :3, 3]

    print(f"=== Robot Path (First 500) vs Sequence 1 Analysis ===")
    
    results = []
    for i in range(len(target_centers)):
        t_c = target_centers[i]
        
        # 计算到 seq1 的所有帧的距离
        dists = np.linalg.norm(seq1_centers - t_c, axis=1)
        min_idx = np.argmin(dists)
        min_dist = dists[min_idx]
        
        results.append({
            "target_idx": int(i),
            "closest_seq1_idx": int(min_idx),
            "dist": float(min_dist)
        })

    # 3. 输出详细列表 (每 10 帧采样显示一下，避免刷屏)
    print(f"{'Target Frame':<15} | {'Closest Seq1':<15} | {'Distance (m)':<15}")
    print("-" * 50)
    for i in range(0, 500, 10): # 每 10 帧打印一行
        r = results[i]
        print(f"{r['target_idx']:04d}           | {r['closest_seq1_idx']:04d}           | {r['dist']:.4f}")

    # 4. 统计信息
    all_dists = [r['dist'] for r in results]
    print("-" * 50)
    print(f"Average Distance: {np.mean(all_dists):.4f}m")
    print(f"Min Distance:     {np.min(all_dists):.4f}m (at target frame {np.argmin(all_dists)})")
    print(f"Max Distance:     {np.max(all_dists):.4f}m (at target frame {np.argmax(all_dists)})")

    # 5. 保存完整结果到 JSON
    output_path = "/workspace_fs/guidedvd-3dgs/seq1_nearest_neighbor.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\n✅ Full results saved to {output_path}")

if __name__ == "__main__":
    analyze_seq1_similarity()
