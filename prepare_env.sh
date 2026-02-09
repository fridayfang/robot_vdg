ln -sf /dataset_rc_b1/chenjiehku/r2s/hg/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth third_party/ViewCrafter/checkpoints/
ln -sf /dataset_rc_b1/chenjiehku/r2s/hg/ViewCrafter_25/model.ckpt third_party/ViewCrafter/checkpoints/

mkdir -p ~/.cache/huggingface/hub/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K/snapshots/default
ln -sfn /dataset_rc_b1/chenjiehku/r2s/hg/CLIP-ViT-H-14-laion2B-s32B-b79K/* ~/.cache/huggingface/hub/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K/snapshots/default/

echo "☁️ 正在生成 DUSt3R 点云..."
python tools/get_replica_dust3r_pcd.py

echo "🏋️ 正在开始 Baseline Pure 3DGS 训练..."
bash scripts/run_replica_baseline.sh replica_baseline 0