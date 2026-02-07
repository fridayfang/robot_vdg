#!/bin/bash
echo "Starting visualization script..."
python3 /workspace_fs/guidedvd-3dgs/visualize_robot_w2cs.py > /workspace_fs/guidedvd-3dgs/vis_log.txt 2>&1
echo "Script finished with exit code $?"
ls -l /workspace_fs/guidedvd-3dgs/robot_traj_vis.png >> /workspace_fs/guidedvd-3dgs/vis_log.txt
