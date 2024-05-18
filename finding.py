import h5py

# 打开 HDF5 文件
file_path = 'data/modelnet40_ply_hdf5_2048/ply_data_train2.h5'
f = h5py.File(file_path, 'r')

# 查看文件中的所有数据集
print("Datasets in the file:")
for name in f:
    print(name)

# 如果有标签数据集，可以读取并查看
if 'label' in f:
    labels = f['label'][:]
    print("Labels:", labels)

# 如果有其他辅助数据集，也可以读取并查看
for name in f:
    if name != 'data':
        print(f"{name}: {f[name][:]}")

# 关闭 HDF5 文件
f.close()
