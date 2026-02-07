import os
import torch
import numpy as np
import torchvision
import copy
import random

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

    def _plan_from_json(self, json_path, num_target_trajs=36, search_range=200):
        """
        修正后的 Task-Specific 逻辑：
        1. 从 JSON 的前 search_range 帧中采样出 num_target_trajs 个目标位姿。
        2. 为每个目标位姿寻找最近的训练视角作为 Anchor。
        3. 在 Anchor 和目标位姿之间插值生成 25 帧的轨迹。
        """
        import json
        print(f"=> [TASK-SPECIFIC] Loading robot path from {json_path}...")
        
        if json_path.endswith('.json'):
            with open(json_path, 'r') as f:
                data = json.load(f)
            w2cs = np.array(data['w2cs_matrices'] if isinstance(data, dict) and 'w2cs_matrices' in data else data)
        else:
            w2cs = []
            with open(json_path, 'r') as f:
                for line in f:
                    nums = [float(x) for x in line.strip().split()]
                    if len(nums) == 16: w2cs.append(np.array(nums).reshape(4, 4))
            w2cs = np.array(w2cs)

        # 1. 确定目标位姿 (Target Poses)
        # 仅取前 200 帧，并从中均匀采样 36 帧
        actual_range = min(len(w2cs), search_range)
        sample_indices = np.linspace(0, actual_range - 1, num_target_trajs, dtype=int)
        target_w2cs = w2cs[sample_indices]
        target_c2ws = [np.linalg.inv(w2c) for w2c in target_w2cs] # 修正了这里的变量名错误

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

        # 3. 逐个目标进行锚定与插值
        print(f"   Generating {num_target_trajs} trajectories by interpolating from nearest train views...")
        for i, target_c2w in enumerate(target_c2ws):
            # 寻找最近的 Anchor View
            target_pos = target_c2w[:3, 3]
            dists = np.linalg.norm(train_c2ws[:, :3, 3] - target_pos, axis=1)
            anchor_idx = np.argmin(dists)
            
            # 使用 ViewCrafter 的插值逻辑生成 25 帧轨迹
            # 我们需要模拟 get_candidate_poses 的输出格式来调用插值
            # 或者直接计算相对变换
            anchor_c2w = train_c2ws[anchor_idx]
            
            # 这里我们利用 vc_wrapper 内部的插值工具
            # 注意：ViewCrafter 的插值通常是在其定义的局部坐标系下进行的
            # 为了简化且保证效果，我们直接在世界坐标系下进行 SLERP 插值
            from scipy.spatial.transform import Rotation as R_tool
            from scipy.spatial.transform import Slerp
            
            times = np.linspace(0, 1, 25)
            key_rots = R_tool.from_matrix([anchor_c2w[:3, :3], target_c2w[:3, :3]])
            key_times = [0, 1]
            slerp = Slerp(key_times, key_rots)
            interp_rots = slerp(times).as_matrix()
            
            interp_poss = np.outer(1 - times, anchor_c2w[:3, 3]) + np.outer(times, target_c2w[:3, 3])
            
            traj_segment = np.zeros((25, 4, 4))
            traj_segment[:, :3, :3] = interp_rots
            traj_segment[:, :3, 3] = interp_poss
            traj_segment[:, 3, 3] = 1.0
            
            # 格式: [索引, 位姿矩阵(25,4,4), 缩放, 尺度索引]
            trajectory_pool[anchor_idx].append([i, traj_segment, 1.0, 1])

        print(f"   Successfully created trajectory pool with {num_target_trajs} path-driven segments.")
        return self._post_process_pool(trajectory_pool)

    def _plan_paper_default(self, fovx, fovy, intrinsic, H, W):
        """实现论文 Sec 3.2 的探测采样逻辑"""
        print("=> Using 'paper_default' sampling method...")
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
        """增强版 3D 可视化：增加相机视锥体 (Frustum) 可视化"""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        def draw_frustum(ax, c2w, color, alpha=0.5, frustum_scale=0.05):
            """绘制相机视锥体的内部辅助函数"""
            # 相机局部坐标系下的 5 个点 (顶点 + 图像平面 4 个角)
            w, h = 0.8, 0.6 # 比例 4:3
            corners = np.array([
                [0, 0, 0],                          # 相机中心 (原点)
                [-w, -h, 1], [w, -h, 1], [w, h, 1], [-w, h, 1] # 图像平面四个角 (Z=1)
            ]) * frustum_scale
            
            # 变换到世界坐标系
            corners_h = np.hstack([corners, np.ones((5, 1))])
            world_corners = (c2w @ corners_h.T).T[:, :3]
            
            # 绘制 4 条从原点发出的侧边
            for i in range(1, 5):
                ax.plot([world_corners[0,0], world_corners[i,0]],
                        [world_corners[0,1], world_corners[i,1]],
                        [world_corners[0,2], world_corners[i,2]], color=color, alpha=alpha, linewidth=0.8)
            
            # 绘制图像平面的矩形框
            rect_idx = [1, 2, 3, 4, 1]
            ax.plot(world_corners[rect_idx, 0], world_corners[rect_idx, 1], world_corners[rect_idx, 2], 
                    color=color, alpha=alpha, linewidth=0.8)

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        colors = plt.cm.get_cmap('tab10', len(trajectory_pool))

        all_pts = []
        for view_idx, trajs in trajectory_pool.items():
            color = colors(int(view_idx))
            for i, (cand_idx, poses, scale, s_idx) in enumerate(trajs):
                centers = poses[:, :3, 3] # [25, 3]
                
                # 1. 绘制轨迹线
                alpha = 0.8 if s_idx == 1 else (0.5 if s_idx == 2 else 0.3)
                ax.plot(centers[:, 0], centers[:, 1], centers[:, 2], color=color, alpha=alpha * 0.5, linewidth=1)
                
                # 2. 绘制起点视锥 (仅为每个 View 的第一条轨迹画一次起点，避免重叠)
                if i == 0:
                    draw_frustum(ax, poses[0], color, alpha=1.0, frustum_scale=0.08)
                    ax.scatter(centers[0,0], centers[0,1], centers[0,2], color=color, s=20, label=f"View {view_idx}")
                
                # 3. 绘制轨迹终点的视锥 (补全的视角)
                draw_frustum(ax, poses[-1], color, alpha=alpha * 0.6, frustum_scale=0.05)
                
                all_pts.append(centers)

        # 坐标轴自动缩放与标签
        if all_pts:
            all_pts = np.concatenate(all_pts, axis=0)
            mid, max_r = (all_pts.max(0) + all_pts.min(0)) / 2, (all_pts.max(0) - all_pts.min(0)).max() / 2
            ax.set_xlim(mid[0]-max_r, mid[0]+max_r); ax.set_ylim(mid[1]-max_r, mid[1]+max_r); ax.set_zlim(mid[2]-max_r, mid[2]+max_r)

        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title("GuidedVD Trajectory Pool (Frustum Visualization)")
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        
        save_path = os.path.join(self.scene.model_path, save_name)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"📊 [锥体可视化] 带有视锥的轨迹图已保存至: {save_path}")

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
        print(f"📊 [相机分布图] 已保存至: {save_path}")

    def _plan_unitree_robot(self, **kwargs):
        """宇树机器人专用采样逻辑（待开发）"""
        print("=> Using 'unitree_robot' kinematics-aware sampling...")
        return {}, {}

    def _post_process_pool(self, trajectory_pool):
        shuffle_pool = copy.deepcopy(trajectory_pool)
        for k in shuffle_pool: random.shuffle(shuffle_pool[k])
        return trajectory_pool, shuffle_pool
