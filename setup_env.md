# guidedvd-3dgs 环境适配说明 (PyTorch 2.1.0 + CUDA 12.1)

本项目原本推荐使用 PyTorch 1.13.1 + CUDA 11.7，但为了保持现有环境稳定性，我们成功在 **PyTorch 2.1.0 + CUDA 12.1** 下完成了适配。

### 1. 核心依赖补全
对比项目 `requirements.txt`，我们在环境中补全了以下库：
*   `decord`: 视频加载
*   `open-clip-torch`: CLIP 模型支持
*   `roma`: 3D 旋转计算
*   `altair`: 声明式统计可视化
*   `ftfy`: 文本编码修复

### 2. 核心算子编译 (Submodules)
项目依赖两个自定义的 CUDA 算子，必须手动编译安装：
*   **simple-knn**: 用于点云的空间索引。
*   **diff-gaussian-rasterization-confidence**: 带有置信度引导的 3DGS 渲染器。

**关键改动**：
在安装时使用了 `--no-build-isolation` 参数。这是因为在 PyTorch 2.1+ 环境下，默认的隔离构建环境可能找不到宿主机的 `torch` 库，导致编译失败。

### 3. 运行前准备
在开始训练前，需要完成以下数据准备（已集成在 `setup_env.sh` 中）：

#### A. 权重文件
权重已自动从以下路径链接到 `third_party/ViewCrafter/checkpoints/`：
*   **DUSt3R**: `/dataset_rc_b1/chenjiehku/r2s/hg/DUSt3R/`
*   **ViewCrafter**: `/dataset_rc_b1/chenjiehku/r2s/hg/ViewCrafter_25/`
*   **CLIP (离线缓存)**: 已链接至 `~/.cache/huggingface/hub/`，支持离线运行。

#### B. 数据集
Replica 数据集已从 `/dataset_rc_b1/chenjiehku/r2s/gs_dataset/Replica/` 自动解压到 `dataset/Replica/`。

### 4. 数据预处理
在正式开始训练之前，必须完成以下数据处理流程（已集成在 `setup_env.sh` 中）：

1.  **Colmap 格式转换** (`tools/replica_to_colmap.py`)：
    *   **作用**：将 Replica 的原始轨迹文件 (`traj_w_c.txt`) 和图像转换为 3DGS 识别的 `sparse/0/` 结构。
    *   **改动**：已将脚本中的 `base_path` 修改为 `/workspace_fs/guidedvd-3dgs/dataset/Replica`。

2.  **DUSt3R 点云生成** (`tools/get_replica_dust3r_pcd.py`)：
    *   **作用**：利用 DUSt3R 模型从稀疏视图中估计初始点云，作为 3DGS 的几何初始化。
    *   **注意**：此步骤会调用 GPU，处理全部 6 个场景大约需要 15 分钟。生成的点云将存放在 `dust3r_results/` 目录下。

### 5. 训练流程与脚本说明
项目 `scripts/` 目录下包含多种实验脚本，其主要区别在于**数据集类型**和**训练策略**：

#### A. Replica 数据集实验 (合成场景)
*   **`run_replica_baseline.sh`**: 
    *   **类型**：基础实验。
    *   **说明**：使用 6 个稀疏视角进行标准 3DGS 训练，不包含视频先验引导。
*   **`run_replica_guidedvd.sh`**: 
    *   **类型**：核心实验。
    *   **说明**：在 Baseline 基础上引入视频扩散模型 (ViewCrafter) 的引导，通过生成伪视图来提升稀疏视角的重建质量。
*   **`run_replica_baseline_with_project_cam.sh`**: 
    *   **类型**：消融/变体实验。
    *   **说明**：在 Baseline 训练中引入投影相机 (Projected Camera) 的逻辑，通常用于测试不同的相机采样策略。
*   **`run_replica_guidedvd_tworenderer.sh`**: 
    *   **类型**：进阶实验。
    *   **说明**：使用双渲染器架构进行 GuidedVD 训练，通常是为了分离引导信号和原始监督信号，以获得更稳定的优化。

#### B. ScanNet++ 数据集实验 (真实场景)
*   **`run_scannetpp_baseline.sh`**: ScanNet++ 的标准 3DGS 基准训练。
*   **`run_scannetpp_guidedvd.sh`**: ScanNet++ 的核心 GuidedVD 引导训练。
*   **`run_scannetpp_guidedvd_hybrid_traj.sh`**: 
    *   **类型**：高级实验。
    *   **说明**：针对真实场景设计的混合轨迹 (Hybrid Trajectory) 引导，结合了多种路径采样方式以应对复杂的室内环境。

### 6. 源码适配与 Bug 修复 (针对 PyTorch 2.1.0)
为了在现有环境下成功运行，我们对 `third_party/ViewCrafter` 进行了深度适配：

#### A. CLIP 文本编码器 MHA 形状修复
*   **文件**：`third_party/ViewCrafter/lvdm/modules/encoders/condition.py`
*   **问题**：PyTorch 2.1.0 的 `MultiHeadAttention` 在处理单条文本（Batch Size=1）时，会误判输入维度并报错 `mask shape [77, 77] but should be (1, 1)`。
*   **修复**：在 `text_transformer_forward` 函数中，针对 `bsz=1` 的情况禁用了显式 Mask 传递，利用 CLIP 内部默认的 Causal Mask 绕过此 Bug。

#### B. VisionTransformer 属性兼容性
*   **文件**：`third_party/ViewCrafter/lvdm/modules/encoders/condition.py`
*   **问题**：新版 `open_clip` 产生的模型对象缺失 `input_patchnorm` 属性。
*   **修复**：添加了 `hasattr` 检查，确保代码在不同版本的 `open_clip` 下都能稳定运行。

#### C. 视频保存的 PyAV 兼容性修复
*   **文件**：`third_party/ViewCrafter/utils_vc/pvd_utils.py`
*   **问题**：新版 `PyAV` 导致 `torchvision.io.write_video` 报错 `TypeError: an integer is required`。
*   **修复**：将视频保存逻辑替换为 `imageio.mimsave`，避免了底层多媒体库的版本冲突。

#### D. 训练进度条 (tqdm) 显示优化
*   **文件**：`train_guidedvd.py`
*   **问题**：视频生成过程（约 5 分钟）会完全阻塞进度条更新，导致日志显示一直卡在 0%。
*   **修复**：将 `tqdm` 改为手动 `update` 模式，并设置 `mininterval=1.0`，确保在视频生成结束后进度条能实时刷新。

### 7. 离线权重加载配置
*   **VGG 权重**：已手动将 `vgg19-dcbb9e9d.pth` 链接至 `~/.cache/torch/hub/checkpoints/`，解决了 LPIPS 损失函数的 Hash 校验失败。
*   **CLIP 权重**：在 `condition.py` 中强制指定本地加载路径，绕过了 Hugging Face 在线下载。
*   **GPU ID**：在脚本中将 `--guidance_gpu_id` 适配为 `0`，以支持单卡环境。

### 8. 验证命令
可以通过以下命令验证环境是否就绪：
```bash
# 验证核心算子导入
python -c "import diff_gaussian_rasterization; import simple_knn; print('OK')"

# 验证训练脚本导入
python -c "from train_baseline import *; from train_guidedvd import *; print('OK')"
```
