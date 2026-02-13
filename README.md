## how to run
bash run_train_eval_baseline.sh

bash run_task_specific_fixed.sh

## 评估
python3 eval_walk_traj.py     --model_path /workspace_fs/robot_vdg/output/replica_office2_eval_0209_0942     --iteration 10000     --w2cs /workspace_fs/robot_walk_2/w2cs.json     --gt_dir /workspace_fs/robot_walk_2/gt_images     --total_frames 300     --num_samples 30

python3 eval_walk_traj.py     --model_path /workspace_fs/robot_vdg/output/replica_office2_eval_0209_0942/baseline_default     --iteration 10000     --w2cs /workspace_fs/robot_walk_2/w2cs.json     --gt_dir /workspace_fs/robot_walk_2/gt_images     --total_frames 300     --num_samples 30