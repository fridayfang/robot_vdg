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

# 4. 启动 Paper Default (Baseline) 训练任务
# 使用 --guidance_random_traj 触发论文默认的全向采样逻辑
echo "=> Launching Paper Default (Baseline) Training..."
python3 train_guidedvd.py \
    -s $DATASET_PATH \
    -m "$OUTPUT_ROOT/baseline_default" \
    --tb_log_dir "$tb_log_dir" \
    --guidance_random_traj \
    --test_indices_file $TEST_SET \
    --iterations 10000 \
    --test_iterations 1000 5000 10000 \
    --guidance_gpu_id 0 \
    --dataset Replica \
    --images rgb \
    --eval

echo "=> Baseline Training Complete."
