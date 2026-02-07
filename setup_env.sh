#!/bin/bash

# ==============================================================================
# guidedvd-3dgs 环境配置脚本
# 基于 PyTorch 2.1.0 + CUDA 12.1 环境适配
# ==============================================================================

# 1. 导出编译加速变量
export MAX_JOBS=4

echo "🚀 开始安装 guidedvd-3dgs 依赖项..."

# 2. 安装缺失的 Python 基础依赖
# 注意：这些包是根据 requirements.txt 对比当前环境后补全的
pip install decord open-clip-torch roma altair ftfy

# 3. 编译并安装核心子模块
# 使用 --no-build-isolation 以确保编译器能直接访问当前环境中的 torch
echo "🛠️ 正在编译 simple-knn..."
cd /workspace_fs/guidedvd-3dgs/submodules/simple-knn && pip install . --no-build-isolation

echo "🛠️ 正在编译 diff-gaussian-rasterization (confidence version)..."
cd /workspace_fs/guidedvd-3dgs/submodules/diff-gaussian-rasterization-confidence && pip install . --no-build-isolation

# 4. 建立权重文件软链接
echo "🔗 正在建立权重文件软链接..."
mkdir -p third_party/ViewCrafter/checkpoints
ln -sf /dataset_rc_b1/chenjiehku/r2s/hg/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth third_party/ViewCrafter/checkpoints/
ln -sf /dataset_rc_b1/chenjiehku/r2s/hg/ViewCrafter_25/model.ckpt third_party/ViewCrafter/checkpoints/

# 4.1 建立 CLIP 权重软链接 (用于离线加载)
mkdir -p ~/.cache/huggingface/hub/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K/snapshots/default
ln -sfn /dataset_rc_b1/chenjiehku/r2s/hg/CLIP-ViT-H-14-laion2B-s32B-b79K/* ~/.cache/huggingface/hub/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K/snapshots/default/
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 5. 准备数据集 (解压 Replica)
echo "📦 正在准备数据集..."
mkdir -p dataset/Replica
cd dataset/Replica
for f in /dataset_rc_b1/chenjiehku/r2s/gs_dataset/Replica/*.zip; do
    base=$(basename "$f" .zip)
    if [ ! -d "$base" ]; then
        echo "Unzipping $base..."
        unzip -q "$f"
    fi
done

# 6. 数据预处理 (Replica)
echo "🔍 正在执行数据预处理..."
# 6.1 转换为 Colmap 格式
python tools/replica_to_colmap.py

# 6.2 生成 DUSt3R 点云 (注意：此步骤需要 GPU，耗时较长)
echo "☁️ 正在生成 DUSt3R 点云..."
python tools/get_replica_dust3r_pcd.py

# 7. 训练 Baseline 3DGS
echo "🏋️ 正在开始 Baseline 3DGS 训练..."
bash scripts/run_replica_baseline.sh replica_baseline 0

# 8. 返回项目根目录
cd /workspace_fs/guidedvd-3dgs

echo "✅ [SUCCESS] 环境配置、数据准备、预处理及 Baseline 训练全部完成！"
