#!/bin/bash

# 设置 PYTHONPATH 确保能找到项目模块
export PYTHONPATH=$PYTHONPATH:/workspace_fs/guidedvd-3dgs

#!/bin/bash

IFS='/' read -ra parts <<<${BIFROST_JOB_DIR}
job_artifacts_dir="/workspace/${parts[-2]}/${parts[-1]}"
tb_log_dir="${job_artifacts_dir}/xflow_logs"


# 1. 设置路径
TIMESTAMP=$(date +"%m%d_%H%M")
PROJECT_ROOT="/workspace_fs/guidedvd-3dgs"
DATASET_PATH="$PROJECT_ROOT/dataset/Replica/office_2/Sequence_2"
TEST_SET="/workspace_fs/test_set.json"
OUTPUT_ROOT="$PROJECT_ROOT/output/replica_office2_eval_${TIMESTAMP}"

# 2. 准备环境
cd $PROJECT_ROOT
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT

# 3. 杀死可能存在的旧进程
pkill -f train_guidedvd.py

# 4. 启动 机器人路径采样 训练任务
# 使用 --robot_traj_path  机器人路径采样
echo "=> Launching Task-Specific (Ours) Training..."
python3 train_guidedvd.py \
    -s $DATASET_PATH \
    -m "$OUTPUT_ROOT/ours_task_specific_fixed" \
    --tb_log_dir "$tb_log_dir" \
    --robot_traj_path /workspace_fs/robot_walk_2/w2cs.json \
    --test_indices_file $TEST_SET \
    --iterations 10000 \
    --test_iterations 10 1000 2000 3000 5000 7000 10000 \
    --guidance_gpu_id 0 \
    --dataset Replica \
    --images rgb \
    --eval

echo "=> Baseline Training Complete."
