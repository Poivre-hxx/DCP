import os
import open3d as o3d

file_path = "D:/HDU/三维点云/DCP/python_qt/data/airplane2.ply"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File {file_path} does not exist.")
else:
    print(f"File {file_path} found.")

    # 加载点云
    pcd = o3d.read_point_cloud(file_path)
    # 确保点云不是空的
    if not pcd.has_points():
        raise ValueError("PointCloud is empty.")
