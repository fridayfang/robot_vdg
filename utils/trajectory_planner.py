import os
import torch
import numpy as np
import torchvision
import copy
import random
from loguru import logger

class TrajectoryPlanner:
    """
    轨迹规划器：封装 GuidedVD 的轨迹初始化策略。
    支持多种采样方法（论文默认、机器人运动学等）、轨迹评估与 3D 可视化。
    """
    def __init__(self, scene, vc_wrapper, easy_renderer, opt, device="cuda"):
        self.scene = scene
        self.vc_wrapper = vc_wrapper
        self.easy_renderer = easy_renderer
        self.opt = opt
        self.device = device
        self.evaluation_results = {} 

    def init_dirs(self):
        """初始化轨迹存储目录"""
        traj_root = self.scene.model_path
        self.save_dirs = {i: os.path.join(traj_root, f"define_traj{'_scale' if i>1 else ''}{'3' if i==3 else ''}/") for i in [1, 2, 3]}
        for d in self.save_dirs.values():
            os.makedirs(d, exist_ok=True)

    def evaluate_trajectory(self, alpha_mask):
        """评估函数：目前根据渲染空洞面积评分"""
        return alpha_mask.view(alpha_mask.shape[0], -1).sum(-1)

    def plan_trajectories(self, method="paper_default", **kwargs):
        """主入口：生成轨迹池"""
        self.init_dirs()
        if method == "paper_default":
            return self._plan_paper_default(**kwargs)
        elif method == "unitree_robot":
            return self._plan_unitree_robot(**kwargs)
        elif method == "from_json":
            return self._plan_from_json(**kwargs)
        raise ValueError(f"Unknown method: {method}")

    def _plan_from_json(self, json_path, num_target_trajs=36, search_range=200, fovx=None, fovy=None, intrinsic=None, H=None, W=None):
        """
        统一 C2W 处理逻辑：
        1. 识别并统一加载为 C2W (Camera-to-World)。
        2. 在 C2W 空间进行多尺度插值。
        3. 生成预览图时转换为 W2C 渲染。
        """
        import json
        logger.info(f"=> [TASK-SPECIFIC] Loading robot path from {json_path}...")
        
        # 加载并统一转换为 C2W
        if json_path.endswith('.json'):
            with open(json_path, 'r') as f:
                data = json.load(f)
            # json 格式明确为 w2cs_matrices (W2C)，需要求逆
            raw_matrices = np.array(data['w2cs_matrices'] if isinstance(data, dict) and 'w2cs_matrices' in data else data)
            target_all_c2ws = [np.linalg.inv(m) for m in raw_matrices]
            logger.info("   Detected .json format: Interpreting as W2C and converting to C2W.")
        else:
            # traj_w_c.txt 实际上是 C2W 格式
            raw_matrices = []
            with open(json_path, 'r') as f:
                for line in f:
                    nums = [float(x) for x in line.strip().split()]
                    if len(nums) == 16: raw_matrices.append(np.array(nums).reshape(4, 4))
            target_all_c2ws = np.array(raw_matrices)
            logger.info("   Detected .txt format: Interpreting as C2W directly.")

        # 1. 采样目标位姿 (C2W)
        actual_range = min(len(target_all_c2ws), search_range)
        sample_indices = np.linspace(0, actual_range - 1, num_target_trajs, dtype=int)
        target_c2ws = [target_all_c2ws[i] for i in sample_indices]

        # 2. 获取训练视角的 C2W 用于锚定
        train_c2ws = []
        train_cams = self.scene.getTrainCameras()
        for cam in train_cams:
            w2c = np.eye(4)
            w2c[:3, :3] = cam.R.transpose()
            w2c[:3, 3] = cam.T
            train_c2ws.append(np.linalg.inv(w2c))
        train_c2ws = np.array(train_c2ws)

        trajectory_pool = {idx: [] for idx in range(len(train_c2ws))}

        # 3. 逐个尺度进行生成 (1.0, 1/3, 1/10)
        configs = [(1.0, 1), (1.0/3.0, 2), (1.0/10.0, 3)] # (factor, s_idx)
        logger.info(f"   Generating multi-scale trajectories (C2W Space) from nearest train views...")
        
        from scipy.spatial.transform import Rotation as R_tool
        from scipy.spatial.transform import Slerp

        for factor, s_idx in configs:
            logger.info(f"   => Processing Scale {s_idx} (factor {factor:.2f})...")
            for i, target_c2w in enumerate(target_c2ws):
                # 寻找最近的 Anchor View (C2W 空间计算距离)
                target_pos = target_c2w[:3, 3]
                dists = np.linalg.norm(train_c2ws[:, :3, 3] - target_pos, axis=1)
                anchor_idx = np.argmin(dists)
                anchor_c2w = train_c2ws[anchor_idx]
                
                # C2W 空间插值
                times = np.linspace(0, factor, 25)
                key_rots = R_tool.from_matrix([anchor_c2w[:3, :3], target_c2w[:3, :3]])
                key_times = [0, 1]
                slerp = Slerp(key_times, key_rots)
                interp_rots = slerp(times).as_matrix()
                interp_poss = np.outer(1 - times, anchor_c2w[:3, 3]) + np.outer(times, target_c2w[:3, 3])
                
                traj_segment = np.zeros((25, 4, 4))
                traj_segment[:, :3, :3] = interp_rots
                traj_segment[:, :3, 3] = interp_poss
                traj_segment[:, 3, 3] = 1.0
                
                # 4. 渲染预览图 (需要 W2C)
                if i < 3 and intrinsic is not None:
                    curr_save_dir = os.path.join(self.save_dirs[s_idx], str(anchor_idx))
                    os.makedirs(curr_save_dir, exist_ok=True)
                    # 渲染终点视角：C2W -> W2C
                    preview_w2c = np.linalg.inv(traj_segment[-1])
                    res, _, _ = self.easy_renderer.render(preview_w2c, intrinsic, H, W)
                    torchvision.utils.save_image(res.clamp(0, 1), os.path.join(curr_save_dir, f"path_{i}_end.png"))

                trajectory_pool[anchor_idx].append([i, traj_segment, factor, s_idx])

        logger.info(f"   Successfully created multi-scale trajectory pool with {num_target_trajs * 3} segments.")
        return self._post_process_pool(trajectory_pool)

    def _plan_paper_default(self, fovx, fovy, intrinsic, H, W):
        """实现论文 Sec 3.2 的探测采样逻辑"""
        logger.info("=> Using 'paper_default' sampling method...")
        trajectory_pool = {}
        mask_thsh = 0.1 * H * W
        d_theta = [-30, -15, 0, 15, 30] if self.opt.guidance_vc_center_scale != 1 else [-15, -7.5, 0, 7.5]
        original_scale = self.vc_wrapper.vc_opts.center_scale

        configs = [(1.0, 3, 1), (1.0/3.0, 2, 2), (1.0/10.0, 1, 3)] # (factor, k, s_idx)

        for train_idx in range(len(self.scene.scene_info_train_indices)):
            select_c2w_trajs = []
            train_cam_id = self.scene.scene_info_train_indices[train_idx]
            
            for factor, k, s_idx in configs:
                self.vc_wrapper.vc_opts.center_scale = original_scale * factor
                curr_save_dir = os.path.join(self.save_dirs[s_idx], str(train_cam_id))
                os.makedirs(curr_save_dir, exist_ok=True)

                c2w_candidates, others = self.vc_wrapper.get_candidate_poses(
                    d_phi=[-30, -15, 0, 15, 30], d_theta=d_theta, fovx=fovx, fovy=fovy, which_train_view=train_idx)
                
                # 评估采样点
                gs_render_alphas = []
                for i, c2w in enumerate(c2w_candidates.cpu().numpy()):
                    res, alpha, _ = self.easy_renderer.render(np.linalg.inv(c2w), intrinsic, H, W)
                    alpha_mask = (alpha.clamp(0, 1) < 0.7).to(torch.float32)
                    gs_render_alphas.append(alpha_mask)
                    if i < 5: # 仅保存前5个预览图以节省空间
                        torchvision.utils.save_image(res.clamp(0, 1), os.path.join(curr_save_dir, f"{i}.png"))

                # 筛选并生成轨迹
                processed_masks = self.vc_wrapper.process_mask(torch.stack(gs_render_alphas, 0))
                scores = self.evaluate_trajectory(processed_masks)
                filtered = (scores < mask_thsh).nonzero(as_tuple=True)[0]
                topk = filtered[torch.argsort(scores[filtered], descending=True)[:k]]
                
                for j in topk:
                    interp = self.vc_wrapper.interpolate_trajectory(others["c2ws"], others["d_phis"][j], others["d_thetas"][j], others["d_rs"][j])
                    traj_world = (torch.bmm(others["transform_back"].unsqueeze(0).expand_as(interp), interp)).cpu().numpy()
                    select_c2w_trajs.append([j.item(), traj_world, self.vc_wrapper.vc_opts.center_scale, s_idx])
            
            trajectory_pool[train_idx] = select_c2w_trajs

        self.vc_wrapper.vc_opts.center_scale = original_scale
        return self._post_process_pool(trajectory_pool)

    def visualize(self, trajectory_pool, save_name="trajectory_vis.png"):
        """增强版 3D 可视化：增加多尺度区分和时间戳验证"""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import time
        
        def draw_frustum(ax, c2w, color, alpha=0.5, frustum_scale=0.05, offset_z=0):
            """绘制相机视锥体"""
            w, h = 0.8, 0.6 
            corners = np.array([
                [0, 0, 0], 
                [-w, -h, 1], [w, -h, 1], [w, h, 1], [-w, h, 1]
            ]) * frustum_scale
            
            corners_h = np.hstack([corners, np.ones((5, 1))])
            world_corners = (c2w @ corners_h.T).T[:, :3]
            world_corners[:, 2] += offset_z # 增加偏移
            
            for i in range(1, 5):
                ax.plot([world_corners[0,0], world_corners[i,0]],
                        [world_corners[0,1], world_corners[i,1]],
                        [world_corners[0,2], world_corners[i,2]], color=color, alpha=alpha, linewidth=0.8)
            rect_idx = [1, 2, 3, 4, 1]
            ax.plot(world_corners[rect_idx, 0], world_corners[rect_idx, 1], world_corners[rect_idx, 2], 
                    color=color, alpha=alpha, linewidth=0.8)

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        colors = plt.cm.get_cmap('tab10', len(trajectory_pool))

        logger.info(f"=> [VISUALIZE] Drawing trajectory pool to {save_name}...")
        
        all_pts = []
        for view_idx, trajs in trajectory_pool.items():
            color = colors(int(view_idx))
            for i, (cand_idx, poses, factor, s_idx) in enumerate(trajs):
                centers = poses[:, :3, 3] # [25, 3]
                
                # 为不同尺度增加微小的 Z 偏移，防止完全重叠
                offset = (s_idx - 1) * 0.05 
                plot_centers = centers.copy()
                plot_centers[:, 2] += offset
                
                # 1. 绘制轨迹线 (不同线型)
                ls = '-' if s_idx == 1 else ('--' if s_idx == 2 else ':')
                alpha = 0.9 if s_idx == 1 else (0.6 if s_idx == 2 else 0.4)
                ax.plot(plot_centers[:, 0], plot_centers[:, 1], plot_centers[:, 2], 
                        color=color, alpha=alpha * 0.5, linewidth=1.5, linestyle=ls)
                
                # 2. 绘制起点视锥
                if i == 0 and s_idx == 1:
                    draw_frustum(ax, poses[0], color, alpha=1.0, frustum_scale=0.1, offset_z=offset)
                    ax.scatter(plot_centers[0,0], plot_centers[0,1], plot_centers[0,2], color=color, s=30)
                
                # 3. 绘制轨迹终点的视锥
                draw_frustum(ax, poses[-1], color, alpha=alpha, frustum_scale=0.06, offset_z=offset)
                all_pts.append(plot_centers)

        if all_pts:
            all_pts = np.concatenate(all_pts, axis=0)
            mid, max_r = (all_pts.max(0) + all_pts.min(0)) / 2, (all_pts.max(0) - all_pts.min(0)).max() / 2
            ax.set_xlim(mid[0]-max_r, mid[0]+max_r); ax.set_ylim(mid[1]-max_r, mid[1]+max_r); ax.set_zlim(mid[2]-max_r, mid[2]+max_r)

        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        # 增加生成时间戳，方便验证图片是否刷新
        gen_time = time.strftime("%H:%M:%S", time.localtime())
        ax.set_title(f"GuidedVD Trajectory Pool (Multi-Scale)\nUpdate Time: {gen_time} | C2W Logic Fixed")
        
        save_path = os.path.join(self.scene.model_path, save_name)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        logger.info(f"📊 [可视化升级] 已保存至: {save_path}，包含多尺度 Z-Offset 和时间戳。")

    def visualize_cameras(self, train_cams, test_cameras, save_name="images/pose_vis.png"):
        """可视化训练和测试相机的分布图"""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        def draw_frustum(ax, R, T, color, alpha=0.5, frustum_scale=0.05):
            # 构造 c2w
            w2c = np.eye(4)
            w2c[:3, :3] = R.T # R is stored transposed in this codebase
            w2c[:3, 3] = T
            c2w = np.linalg.inv(w2c)
            
            w, h = 0.8, 0.6
            corners = np.array([[0,0,0], [-w,-h,1], [w,-h,1], [w,h,1], [-w,h,1]]) * frustum_scale
            corners_h = np.hstack([corners, np.ones((5, 1))])
            world_corners = (c2w @ corners_h.T).T[:, :3]
            for i in range(1, 5):
                ax.plot([world_corners[0,0], world_corners[i,0]], [world_corners[0,1], world_corners[i,1]], [world_corners[0,2], world_corners[i,2]], color=color, alpha=alpha, linewidth=1)
            rect_idx = [1, 2, 3, 4, 1]
            ax.plot(world_corners[rect_idx, 0], world_corners[rect_idx, 1], world_corners[rect_idx, 2], color=color, alpha=alpha, linewidth=1)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        all_centers = []
        # 1. 绘制训练相机 (绿色)
        for cam in train_cams:
            draw_frustum(ax, cam.R, cam.T, color='green', alpha=1.0, frustum_scale=0.1)
            center = np.linalg.inv(np.vstack([np.hstack([cam.R.T, cam.T.reshape(3,1)]), [0,0,0,1]]))[:3, 3]
            all_centers.append(center)
        ax.scatter([], [], [], color='green', label='Train Cameras') # 仅用于图例

        # 2. 绘制测试相机 (红色)
        for cam in test_cameras:
            draw_frustum(ax, cam.R, cam.T, color='red', alpha=0.4, frustum_scale=0.06)
            center = np.linalg.inv(np.vstack([np.hstack([cam.R.T, cam.T.reshape(3,1)]), [0,0,0,1]]))[:3, 3]
            all_centers.append(center)
        ax.scatter([], [], [], color='red', label='Test Cameras') # 仅用于图例

        if all_centers:
            all_centers = np.array(all_centers)
            mid, max_r = (all_centers.max(0) + all_centers.min(0)) / 2, (all_centers.max(0) - all_centers.min(0)).max() / 2
            ax.set_xlim(mid[0]-max_r, mid[0]+max_r); ax.set_ylim(mid[1]-max_r, mid[1]+max_r); ax.set_zlim(mid[2]-max_r, mid[2]+max_r)

        ax.set_title("Train (Green) vs Test (Red) Camera Poses")
        ax.legend()
        save_path = os.path.join(self.scene.model_path, save_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150); plt.close()
        logger.info(f"📊 [相机分布图] 已保存至: {save_path}")

    def _plan_unitree_robot(self, **kwargs):
        """宇树机器人专用采样逻辑（待开发）"""
        logger.info("=> Using 'unitree_robot' kinematics-aware sampling...")
        return {}, {}

    def _post_process_pool(self, trajectory_pool):
        shuffle_pool = copy.deepcopy(trajectory_pool)
        for k in shuffle_pool: random.shuffle(shuffle_pool[k])
        return trajectory_pool, shuffle_pool
