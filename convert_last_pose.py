import json
import numpy as np
from scipy.spatial.transform import Rotation as R

def convert_w2c_to_pos_rot(w2c_matrix):
    # 确保是 numpy 数组
    w2c = np.array(w2c_matrix)
    
    # 1. 计算 C2W 矩阵 (求逆)
    # 对于刚体变换矩阵，逆矩阵可以用更高效的方法，但 np.linalg.inv 最通用
    try:
        c2w = np.linalg.inv(w2c)
    except np.linalg.LinAlgError:
        # 如果最后一行不是 [0,0,0,1]，手动处理
        res = np.eye(4)
        res[:3, :3] = w2c[:3, :3].T
        res[:3, 3] = -w2c[:3, :3].T @ w2c[:3, 3]
        c2w = res

    # 2. 提取位置 (Translation)
    pos = c2w[:3, 3].tolist()
    
    # 3. 提取旋转并转为四元数 (x, y, z, w)
    # 注意：scipy 的 as_quat() 默认返回的就是 [x, y, z, w]
    rot_matrix = c2w[:3, :3]
    rotation = R.from_matrix(rot_matrix)
    rot = rotation.as_quat().tolist()
    
    return {
        "pos": [round(x, 4) for x in pos],
        "rot": [round(x, 4) for x in rot]
    }

# 最后一帧的数据
last_frame_w2c = [
    [-0.6351709961891174, -0.771921694278717, -0.02634766884148121, -2.390352725982666],
  [0.012212919071316719, 0.02407095767557621, -0.9996355175971985, 0.2046399712562561],
  [0.7722745537757874, -0.6352614760398865, -0.005861555226147175, 0.2499568909406662],
  [-2.2038120661704852e-08, 2.9224795028426342e-08, -2.897570516857684e-10, 0.9999999403953552]
]

result = convert_w2c_to_pos_rot(last_frame_w2c)
print(json.dumps(result, indent=2))
