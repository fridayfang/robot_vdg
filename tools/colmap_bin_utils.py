import struct
import numpy as np
import os

def write_cameras_bin(intrinsics, sparse_path, H, W):
    """
    Export cameras to cameras.bin
    MODEL: PINHOLE (id 1), PARAMS: fx, fy, cx, cy
    """
    cameras_bin_file = os.path.join(sparse_path, 'cameras.bin')
    with open(cameras_bin_file, 'wb') as f:
        # Number of cameras (uint64)
        f.write(struct.pack("<Q", len(intrinsics)))
        for i, intrinsic in enumerate(intrinsics):
            # camera_id (int32)
            # model_id (int, PINHOLE is 1)
            # width (uint64)
            # height (uint64)
            # params (double * 4: fx, fy, cx, cy)
            f.write(struct.pack("<iiQQ", i, 1, W, H))
            f.write(struct.pack("<dddd", intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]))

def write_images_bin(world2cam, sparse_path):
    """
    Export images to images.bin
    Format: image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name, points2D
    """
    from tools.replica_to_colmap import rotmat2qvec
    images_bin_file = os.path.join(sparse_path, 'images.bin')
    with open(images_bin_file, 'wb') as f:
        # Number of images (uint64)
        f.write(struct.pack("<Q", world2cam.shape[0]))
        for i in range(world2cam.shape[0]):
            rotation_matrix = world2cam[i, :3, :3]
            qw, qx, qy, qz = rotmat2qvec(rotation_matrix)
            tx, ty, tz = world2cam[i, :3, 3]
            
            # image_id (int32)
            # qvec (double * 4)
            # tvec (double * 3)
            # camera_id (int32)
            # name (string, null-terminated)
            image_name = f"{i}.png\0"
            f.write(struct.pack("<i", i))
            f.write(struct.pack("<dddd", qw, qx, qy, qz))
            f.write(struct.pack("<ddd", tx, ty, tz))
            f.write(struct.pack("<i", i))
            f.write(image_name.encode('utf-8'))
            
            # Number of 2D points (uint64) - set to 0
            f.write(struct.pack("<Q", 0))

def write_points3D_bin(sparse_path):
    """
    Export empty points3D.bin (uint64 0 for number of points)
    """
    points3D_bin_file = os.path.join(sparse_path, 'points3D.bin')
    with open(points3D_bin_file, 'wb') as f:
        f.write(struct.pack("<Q", 0))
