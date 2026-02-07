import os
import requests
from tqdm import tqdm

def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    t = tqdm(total=total_size, unit='iB', unit_scale=True, desc=filename)
    with open(filename, 'wb') as f:
        for data in response.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()

if __name__ == "__main__":
    # Replica office_2 mesh 的公共下载链接 (来自 Semantic-NeRF/Replica 官方公开资源)
    # 注意：如果链接失效，可能需要从官方 Dropbox 镜像获取
    mesh_url = "https://github.com/facebookresearch/Replica-Dataset/raw/master/meshes/office_2.ply"
    
    output_dir = "dataset/Replica/meshes"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "office_2.ply")
    
    print(f"开始下载 Replica office_2 Mesh 到 {output_path}...")
    try:
        # 尝试从 GitHub 镜像下载 (通常较小，仅作为示例，Replica 完整 mesh 较大)
        # 实际上 Replica 的完整 Mesh 通常在 Dropbox 镜像中
        # 这里我们先尝试一个已知的公开链接
        download_file(mesh_url, output_path)
        print("\n✅ 下载完成！")
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        print("建议手动从以下地址下载并放入 dataset/Replica/meshes/office_2.ply :")
        print("https://www.dropbox.com/scl/fo/puh6djua6ewgs0afsswmz/AHiqYQQv7ydbWMcAULTZk1w/Replica_Dataset?dl=0")

