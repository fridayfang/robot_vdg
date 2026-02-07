import trimesh
import sys
import os

def convert_ply_to_stl(ply_path, stl_path):
    if not os.path.exists(ply_path):
        print(f"Error: File {ply_path} does not exist.")
        return

    print(f"Loading {ply_path}...")
    try:
        # Load the mesh
        mesh = trimesh.load(ply_path)
        
        # Check if it's a point cloud or a mesh
        if isinstance(mesh, trimesh.PointCloud):
            print("Error: The input PLY is a PointCloud, not a Mesh. STL requires face information.")
            return

        print(f"Exporting to {stl_path}...")
        # Export as STL
        mesh.export(stl_path)
        print("✅ Conversion successful!")
        
    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    input_ply = "/workspace_fs/guidedvd-3dgs/dataset/Replica/office_2/mesh.ply"
    output_stl = "/workspace_fs/guidedvd-3dgs/dataset/Replica/office_2/mesh.stl"
    convert_ply_to_stl(input_ply, output_stl)
