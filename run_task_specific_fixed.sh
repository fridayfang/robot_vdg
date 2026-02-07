#!/bin/bash

# 设置 PYTHONPATH 确保能找到项目模块
export PYTHONPATH=$PYTHONPATH:/workspace_fs/guidedvd-3dgs

# 运行 Task-Specific 引导训练，并使用固定的测试集索引
python3 train_guidedvd.py \
    -s /workspace_fs/guidedvd-3dgs/dataset/Replica/office_2/Sequence_2 \
    -m /workspace_fs/guidedvd-3dgs/output/replica_guidedvd_office2_task_specific_fixed_test \
    --robot_traj_path /workspace_fs/robot_walk_2/w2cs.json \
    --iterations 10000 \
    --test_iterations 1000 2000 3000 5000 7000 10000 \
    --fixed_test_indices 150 151 152 138 136 137 139 149 153 135 134 140 148 154 133 141 147 155 132 142 \
    --guidance_gpu_id 0 \
    --dataset Replica \
    --images rgb \
    --eval
