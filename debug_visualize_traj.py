import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys

def visualize_trajectories(json_path, output_path="traj_vis.png"):
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 颜色映射：每个训练视点一个颜色
    colors = plt.cm.get_cmap('tab10', len(data))

    for view_idx, trajectories in data.items():
        color = colors(int(view_idx))
        
        # 绘制该视点的所有轨迹
        for traj_idx, (cand_idx, poses_list, scale, s_idx) in enumerate(trajectories):
            poses = np.array(poses_list) # [25, 4, 4]
            centers = poses[:, :3, 3]    # 提取相机中心坐标 [25, 3]
            
            # 1. 绘制轨迹线
            ax.plot(centers[:, 0], centers[:, 1], centers[:, 2], 
                    color=color, alpha=0.6, linewidth=1.5,
                    label=f"View {view_idx}" if traj_idx == 0 else "")
            
            # 2. 绘制起点 (训练相机位置)
            ax.scatter(centers[0, 0], centers[0, 1], centers[0, 2], 
                       color=color, s=50, marker='o')
            
            # 3. 绘制终点 (探测相机位置)
            ax.scatter(centers[-1, 0], centers[-1, 1], centers[-1, 2], 
                       color=color, s=30, marker='x')
            
            # 4. 绘制相机朝向 (取中间一帧作为代表)
            mid_idx = 12
            z_axis = poses[mid_idx, :3, 2] # 相机正前方是 Z 轴
            ax.quiver(centers[mid_idx, 0], centers[mid_idx, 1], centers[mid_idx, 2],
                      z_axis[0], z_axis[1], z_axis[2], 
                      length=0.1, color=color, alpha=0.3)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('GuidedVD Trajectory Pool Visualization')
    
    # 强制等比例缩放，避免坐标轴拉伸
    all_centers = []
    for v in data.values():
        for t in v:
            all_centers.append(np.array(t[1])[:, :3, 3])
    if all_centers:
        all_centers = np.concatenate(all_centers, axis=0)
        max_range = np.array([all_centers[:,0].max()-all_centers[:,0].min(), 
                             all_centers[:,1].max()-all_centers[:,1].min(), 
                             all_centers[:,2].max()-all_centers[:,2].min()]).max() / 2.0
        mid_x = (all_centers[:,0].max()+all_centers[:,0].min()) * 0.5
        mid_y = (all_centers[:,1].max()+all_centers[:,1].min()) * 0.5
        mid_z = (all_centers[:,2].max()+all_centers[:,2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # 仅显示主视图的图例
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1.05, 1))

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"✅ 可视化结果已保存至: {output_path}")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "output/replica_guidedvd_office2/office_2/Sequence_2/traj_debug_data.json"
    visualize_trajectories(path)
