import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def draw_frustum(ax, c2w, color='blue', focal=0.5, width=0.4, height=0.3, alpha=0.3):
    """
    Draws a camera frustum in 3D.
    c2w: 4x4 Camera-to-World matrix
    """
    # Define frustum corners in camera coordinates (z forward)
    # Origin
    origin = np.array([0, 0, 0, 1])
    # Four corners of the image plane at distance 'focal'
    p1 = np.array([-width/2, -height/2, focal, 1])
    p2 = np.array([ width/2, -height/2, focal, 1])
    p3 = np.array([ width/2,  height/2, focal, 1])
    p4 = np.array([-width/2,  height/2, focal, 1])

    # Transform to world coordinates
    points = np.stack([origin, p1, p2, p3, p4])
    points_world = (c2w @ points.T).T[:, :3]

    # Draw lines from origin to corners
    for i in range(1, 5):
        ax.plot([points_world[0, 0], points_world[i, 0]],
                [points_world[0, 1], points_world[i, 1]],
                [points_world[0, 2], points_world[i, 2]], color=color, alpha=alpha)

    # Draw image plane rectangle
    idx = [1, 2, 3, 4, 1]
    ax.plot(points_world[idx, 0], points_world[idx, 1], points_world[idx, 2], color=color, alpha=alpha)

def visualize_w2cs(json_path, output_path="robot_traj_vis.png", sample_rate=10):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    w2cs = np.array(data['w2cs_matrices'])
    print(f"Loaded {len(w2cs)} poses.")

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    all_positions = []

    # Iterate through poses with a sample rate to avoid overcrowding
    for i in range(0, len(w2cs), sample_rate):
        w2c = w2cs[i]
        try:
            c2w = np.linalg.inv(w2c)
        except np.linalg.LinAlgError:
            print(f"Skipping singular matrix at index {i}")
            continue
            
        pos = c2w[:3, 3]
        all_positions.append(pos)
        
        # Color gradient based on time
        color = plt.cm.viridis(i / len(w2cs))
        
        # Draw frustum
        draw_frustum(ax, c2w, color=color, focal=0.1, width=0.08, height=0.06, alpha=0.5)
        
        # Draw a small point at camera origin
        ax.scatter(pos[0], pos[1], pos[2], color=color, s=10)

    # Plot the path line
    all_positions = np.array(all_positions)
    ax.plot(all_positions[:, 0], all_positions[:, 1], all_positions[:, 2], color='gray', linestyle='--', alpha=0.5)

    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title('Robot Trajectory Visualization (W2C to C2W)')
    
    # Set equal aspect ratio
    max_range = np.array([all_positions[:,0].max()-all_positions[:,0].min(), 
                          all_positions[:,1].max()-all_positions[:,1].min(), 
                          all_positions[:,2].max()-all_positions[:,2].min()]).max() / 2.0

    mid_x = (all_positions[:,0].max()+all_positions[:,0].min()) * 0.5
    mid_y = (all_positions[:,1].max()+all_positions[:,1].min()) * 0.5
    mid_z = (all_positions[:,2].max()+all_positions[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.savefig(output_path)
    print(f"✅ Visualization saved to {output_path}")

if __name__ == "__main__":
    visualize_w2cs("/workspace_fs/guidedvd-3dgs/w2cs.json")
