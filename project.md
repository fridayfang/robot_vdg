# Project: Kinematics-Aware Active Scene Completion for Humanoid Robots (Active-GS)

## 1. 研究目标 (Research Objectives)
针对**宇树人形机器人 (Unitree H1/G1)** 在稀疏观测环境下的视角外插（Extrapolation）难题，提出一种**主动感知采样（Active Guidance Sampling）**策略。通过将机器人的**运动学约束（Motion Policy）**注入视频扩散模型（ViewCrafter），构建一个物理一致、语义连贯的 3DGS 仿真环境，旨在解决机器人运动空间与拍摄空间不一致导致的导航失效问题。

---

## 2. 实验设计 (Experimental Setup)

### 2.1 数据集配置
*   **基准数据集 (Benchmark)**: ScanNet++ (精选 3 个复杂室内场景)，利用其高精度 DSLR 真值进行闭环验证。
*   **实战数据集 (Real-world)**: 自采场景 (3 个：实验室、狭窄走廊、杂乱地面)，验证泛化性。
*   **输入限制**: 每个场景仅提供 **6 张** 稀疏视角图片作为初始观测。

### 2.2 核心实验矩阵
| 实验编号 | 实验名称 | 核心操作 | 验证目的 |
| :--- | :--- | :--- | :--- |
| **Exp.1** | **Sim-to-Sim 闭环验证** | 留出法：剔除机器人路径上的真值图，用 Active-GS 脑补后对比。 | 验证路径上的视觉重建精度。 |
| **Exp.2** | **VLA 决策一致性测试** | 将渲染图输入 **OpenVLA**，对比其动作输出与真值图输出的一致性。 | 验证脑补画面的语义真实性。 |
| **Exp.3** | **运动学一致性分析** | 模拟宇树机器人行走时的俯仰（Pitch）与震荡，测试渲染稳定性。 | 验证环境是否符合物理运动逻辑。 |
| **Exp.4** | **消融实验** | 拆解 Planner 引导、不确定性权重、运动学约束（步态/俯仰）等模块。 | 证明各组件对效率和质量的贡献。 |

---

## 3. 指标形式与预估结果 (Metrics & Expected Results)

### 3.1 视觉与几何维度 (Visual & Geometric)
| 指标名称 | 定义/形式 | 预估结果 (Baseline vs. Ours) |
| :--- | :--- | :--- |
| **Path-PSNR** | 机器人运动轨迹附近的平均 PSNR | 18.5dB $\rightarrow$ **24.2dB** (提升 ~30%) |
| **Path-LPIPS** | 路径图像的感知相似度 (越低越好) | 0.25 $\rightarrow$ **0.12** (感知质量翻倍) |
| **Depth-MAE** | 渲染深度与真值 Mesh 深度的平均误差 | 0.25m $\rightarrow$ **0.08m** (几何更贴合物理) |

### 3.2 具身决策与运动维度 (Embodied & Motion)
| 指标名称 | 定义/形式 | 预估结果 (Baseline vs. Ours) |
| :--- | :--- | :--- |
| **Action Agreement** | VLA 动作输出与真值动作的重合率 | 65% $\rightarrow$ **88%** (决策更可靠) |
| **Success Rate** | 50 步连续避障导航的成功率 | 40% $\rightarrow$ **82%** (显著减少撞墙/幻觉) |
| **Jerk (Smoothness)** | 视觉引导下机器人控制指令的加加速度 | 显著降低，消除视角外插导致的控制震荡。 |

### 3.3 效率维度 (Efficiency)
| 指标名称 | 定义/形式 | 预估结果 (Baseline vs. Ours) |
| :--- | :--- | :--- |
| **Sampling Efficiency** | 达到目标 PSNR 所需的 ViewCrafter 调用次数 | **减少 60%** (算力集中在作业空间) |
| **Build Time** | 构建单个可用仿真环境的总耗时 | 相比全场景补全，耗时缩短约 **1/2**。 |

---

## 4. 关键技术亮点 (Selling Points)
1.  **任务导向重建 (Task-Oriented)**: 突破“全场景重建”传统范式，提出为机器人作业空间（Workspace）定制精度。
2.  **运动学闭环 (Kinematics-in-the-loop)**: 首次将人形机器人的俯仰、步态晃动等物理特性引入 3DGS 脑补过程。
3.  **VLA 适配仿真 (VLA-Ready)**: 证明生成式 3D 环境能直接提升视觉语言动作模型在未知区域的 Zero-shot 表现。

---

## 5. 待办事项 (To-Do List)
- [ ] 编写 `trajectory_generator.py`：注入宇树机器人运动学约束位姿。
- [ ] 准备 ScanNet++ 场景 `b20a261f6a` 的位姿筛选脚本。
- [ ] 部署 OpenVLA 推理环境。
- [ ] 完成自采场景的 COLMAP 重建。
