import cv2
import os
import numpy as np
import argparse

def create_comparison_grid(gt_dir, render_dir, output_path, num_images=5, real_img_dir=None, metrics_list=None):
    # 获取所有 gt_*.png 和 render_*.png 文件
    all_files = sorted(os.listdir(gt_dir))
    gt_files = [f for f in all_files if f.startswith('gt_') and f.endswith('.png')]
    render_files = [f for f in all_files if f.startswith('render_') and f.endswith('.png')]
    
    if not gt_files or not render_files:
        # 退回到之前的逻辑
        gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith('.png')])
        render_files = sorted([f for f in os.listdir(render_dir) if f.endswith('.png')])
    
    if not gt_files or not render_files:
        print(f"Error: No images found in {gt_dir} or {render_dir}")
        return

    display_indices = range(min(num_images, len(gt_files), len(render_files)))
    
    rows = []
    for i in display_indices:
        gt_f = gt_files[i]
        # 寻找对应的 render 文件 (假设后缀索引一致)
        idx_str = gt_f.replace('gt_', '').replace('.png', '')
        render_f = f"render_{idx_str}.png"
        
        gt_path = os.path.join(gt_dir, gt_f)
        render_path = os.path.join(render_dir, render_f)
        
        if not os.path.exists(render_path):
            print(f"Warning: Render file {render_path} not found for {gt_f}")
            continue

        gt_img = cv2.imread(gt_path)
        render_img = cv2.imread(render_path)
        
        if gt_img is None or render_img is None: continue
        
        render_img = cv2.resize(render_img, (gt_img.shape[1], gt_img.shape[0]))
        
        # 基础拼接
        combined = np.hstack([gt_img, render_img])

        # 在图片上绘制 PSNR
        if metrics_list and i < len(metrics_list):
            psnr_val = metrics_list[i]
            cv2.putText(combined, f"PSNR: {psnr_val:.2f}dB", (gt_img.shape[1] + 10, gt_img.shape[0] - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.putText(combined, f"GT | {gt_f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(combined, "Render", (gt_img.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        rows.append(combined)
    
    if rows:
        final_grid = np.vstack(rows)
        cv2.imwrite(output_path, final_grid)
        print(f"✅ Comparison grid with PSNR saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", default="/workspace_fs/guidedvd-3dgs/baseline_result/gt")
    parser.add_argument("--render_dir", default="/workspace_fs/guidedvd-3dgs/baseline_result/renders_sampled")
    parser.add_argument("--real_img_dir", default=None)
    parser.add_argument("--output", default="/workspace_fs/baseline_vs_gt.png")
    parser.add_argument("--num_images", type=int, default=5)
    parser.add_argument("--json_metrics", default=None, help="Path to results.json to extract PSNRs")
    args = parser.parse_args()

    metrics_list = None
    if args.json_metrics and os.path.exists(args.json_metrics):
        import json
        with open(args.json_metrics, 'r') as f:
            # 注意：这里假设 json 里的 psnr 是一个列表，对应每一帧
            # 如果 eval_new_path.py 还没保存每帧的列表，我们需要修改它
            data = json.load(f)
            if 'per_frame_psnr' in data:
                metrics_list = data['per_frame_psnr']

    create_comparison_grid(args.gt_dir, args.render_dir, args.output, args.num_images, args.real_img_dir, metrics_list)
