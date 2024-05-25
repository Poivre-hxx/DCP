import numpy as np
import time
import icp
import h5py
import open3d as o3d
import copy

# Constants
num_tests = 50  # number of test iterations
noise_sigma = .06  # 增加噪声的标准差（增加到 0.1）
translation = 2  # max translation of the test set
rotation = np.pi / 3  # max rotation (radians) of the test set

def rotation_matrix(axis, theta):
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.)
    b, c, d = -axis * np.sin(theta / 2.)
    return np.array([[a*a + b*b - c*c - d*d, 2*(b*c - a*d), 2*(b*d + a*c)],
                     [2*(b*c + a*d), a*a + c*c - b*b - d*d, 2*(c*d - a*b)],
                     [2*(b*d - a*c), 2*(c*d + a*b), a*a + d*d - b*b - c*c]])

def load_modelnet_data(file_path):
    with h5py.File(file_path, 'r') as f:
        data = f['data'][:]
        labels = f['label'][:]
    return data, labels

def draw_registration_result(source, target, transformation):
    if source is None or target is None:
        print("Error: source or target is None.")
        return

    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)

    if source_temp is None or target_temp is None:
        print("Error: deepcopy failed.")
        return

    try:
        source_temp.paint_uniform_color([0, 1, 0])  # Green for source point cloud
        transformed_source_temp = copy.deepcopy(source_temp)
        transformed_source_temp.transform(transformation)
        transformed_source_temp.paint_uniform_color([1, 0, 0])  # Red for registered point cloud
        target_temp.paint_uniform_color([0, 0, 1])  # Blue for target point cloud
        o3d.draw_geometries([source_temp, transformed_source_temp, target_temp])
    except Exception as e:
        print(f"Error during visualization: {e}")

def numpy_to_open3d_pc(points):
    if points is None:
        print("Error: input points are None.")
        return None

    pcd = o3d.PointCloud()
    pcd.points = o3d.Vector3dVector(points)

    if not pcd.has_points():
        print("Error: point cloud has no points.")
        return None

    return pcd

def calculate_registration_error_percentage(src, target):
    distances = np.linalg.norm(src - target, axis=1)
    mean_distance = np.mean(distances)
    max_possible_distance = np.sqrt(3)  # 对于单位立方体，最大距离为 sqrt(3)
    error_percentage = (mean_distance / max_possible_distance) * 100
    return error_percentage

def test_icp(modelnet_file_path):
    try:
        # Load ModelNet dataset
        data, labels = load_modelnet_data(modelnet_file_path)
        print("Loaded ModelNet data successfully")

        total_time = 0
        total_error_percentage = 0
        N, dim = data.shape[1], data.shape[2]

        for i in range(num_tests):
            # Randomly select a point cloud from the dataset
            index = 0  # 使用第一个点云
            A = data[index]

            # Generate B by translating, rotating, and adding noise to A
            B = np.copy(A)

            # Translate
            t = np.random.rand(dim) * translation  # 随机平移向量
            B += t  # 对 B 进行平移

            # Rotate
            R = rotation_matrix(np.random.rand(dim), np.random.rand() * rotation)  # 随机旋转矩阵
            B = np.dot(R, B.T).T  # 对 B 进行旋转

            # Add noise
            B += np.random.randn(N, dim) * noise_sigma  # 对 B 添加噪声

            # Shuffle to disrupt correspondence
            np.random.shuffle(B)  # 打乱 B 中点的顺序

            # Convert numpy arrays to Open3D point clouds
            source = numpy_to_open3d_pc(B)
            target = numpy_to_open3d_pc(A)

            if source is None or target is None:
                print("Error: source or target point cloud is None.")
                continue

            # Run ICP with a maximum number of iterations
            start = time.time()
            try:
                T, distances, iterations = icp.icp(B, A, tolerance=0.000001, max_iterations=50)  # 使用 ICP 算法配准 B 和 A
            except Exception as e:
                print(f"Error during ICP iteration {i+1}: {e}")
                continue

            elapsed_time = time.time() - start
            total_time += elapsed_time

            # Calculate mean error for this iteration
            error_percentage = calculate_registration_error_percentage(np.dot(T[:3, :3], B.T).T + T[:3, 3], A)
            total_error_percentage += error_percentage

            # Debug output
            # print(f"Iteration {i + 1} completed in {elapsed_time:.3f} seconds with error percentage {error_percentage:.2f}%")

            # 断言检查
            try:
                assert np.mean(distances) < 6 * noise_sigma  # 检查平均误差是否小于 6 倍噪声标准差
                assert np.allclose(T[0:3, 0:3].T, R, atol=6 * noise_sigma)  # 检查旋转矩阵是否反向
                assert np.allclose(-T[0:3, 3], t, atol=6 * noise_sigma)  # 检查平移向量是否反向
            except AssertionError as e:
                print(f"Assertion error during iteration {i + 1}: {e}")

        print('icp time: {:.3}'.format(total_time / num_tests))  # 打印平均配准时间
        print('average registration error: {:.3}'.format(total_error_percentage / num_tests))  # 打印平均配准误差

        # Draw the results for the first test
        draw_registration_result(source, target, T)

    except Exception as e:
        print(f"Error during ICP testing: {e}")

if __name__ == "__main__":
    modelnet_file_path = 'data/modelnet40_ply_hdf5_2048/ply_data_train2.h5'
    test_icp(modelnet_file_path)
