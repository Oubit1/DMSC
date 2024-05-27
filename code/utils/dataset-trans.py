import torch
import numpy as np
import h5py
import os
from pathlib import Path
import torchvision.transforms as transforms
import cv2

# 读取compare.txt文件中的文件名列表
root1 = '/remote-home/wjh/project/CC-Net-main-data-consist1/data/LA/Reconsitution/2018LA_Seg_Training Set'
compare_file = '/remote-home/wjh/project/CC-Net-main-data-consist1/data/LA/Reconsitution/compare.txt'
with open(compare_file, 'r') as file:
    file_names = [line.strip() for line in file]

# 设置随机种子以保证每次运行结果一致性
np.random.seed(0)

# 重构输出目录
output_path = '/remote-home/wjh/project/CC-Net-main-data-consist1/data/LA/Reconsition_1/2018LA_Seg_Training Set'


for file_name in file_names:
    # 读取第一个目录下的_mri_norm2.h5文件
    file_path_1 = f'/remote-home/wjh/project/CC-Net-main-data-consist1/data/LA/Reconsitution/2018LA_Seg_Training Set/{file_name}/mri_norm2.h5'
    with h5py.File(file_path_1, 'r') as h5f_1:
        keys = list(h5f_1.keys())
        print(keys)
        image_dataset_1 = h5f_1['volume']
        image_dataset_1 = np.transpose(image_dataset_1, (2, 1, 0))
        image_array_1 = image_dataset_1[...]
        image_tensor_1 = torch.from_numpy(image_array_1)
        tensor_size1 = image_tensor_1.size()
        print(tensor_size1)  # 输出：torch.Size([N, C, H, W])
        print('image_tensor_1:', image_tensor_1.shape)  # 添加这行打印语句
        print('image_dataset_1:', image_dataset_1.shape)  # 添加这行打印语句

    # 读取第二个目录下的_mri_norm2.h5文件
    file_path_2 = f'/remote-home/wjh/project/CC-Net-main-data-consist1/data/LA/2018LA_Seg_Training Set/{file_name}/mri_norm2.h5'
    with h5py.File(file_path_2, 'r') as h5f_2:
        image_dataset_2 = h5f_2['image']
        image_array_2 = image_dataset_2[...]
        image_tensor_2 = torch.from_numpy(image_array_2)
        tensor_size2 = image_tensor_2.size()
        print(tensor_size2)  # 输出：torch.Size([N, C, H, W])
        print('image_dataset_2:', image_dataset_2.shape)  # 添加这行打印语句
        print('image_tensor_2:', image_tensor_2.shape)  # 添加这行打印语句

        # 获取数据的深度
        depth = image_dataset_2.shape[2]
        # 裁剪张量以减少总元素数并调整形状
        cropped_tensor = image_tensor_1[:image_tensor_2.size(0), :image_tensor_2.size(1), :depth]
        print('reshaped_tensor1:', cropped_tensor.shape)  # 添加这行打印语句
        # 创建空的重构数据张量
        reconstructed_data = np.zeros((image_tensor_2.shape[0], image_tensor_2.shape[1], depth))
        print('reconstructed_data:', reconstructed_data.shape)
        # 逐层重构数据
        resized_slice_1_flat = None
        resized_slice_2_flat = None
        for i in range(depth):
            # 获取当前切片
            slice_1 = cropped_tensor[:, :, i]
            slice_2 = image_tensor_2[:, :, i]
            # print(slice_1.shape)  # 添加这行打印语句
            # print(slice_2.shape)  # 添加这行打印语句
            if i == 0:
                # 重构数据集首张切片随机从两个数据集中选取
                choices = np.concatenate((slice_1, slice_2), axis=None)
                selected_slice = np.random.choice(choices)
                reconstructed_data[:, :, 0] = torch.from_numpy(np.array(selected_slice))
                continue
            width_reconstructed = image_tensor_2[:, :, i-1].size(1)
            height_reconstructed = image_tensor_2[:, :, i-1].size(0)
            print("width_reconstructed:",  width_reconstructed)
            print("height_reconstructed:", height_reconstructed)
            # 将调整后的切片存入重构数据张量中
            slice_11 = np.array(slice_1)
            slice_22 = np.array(slice_2)
            # 调整slice_1的尺寸
            resized_slice_1 = cv2.resize(slice_11, (width_reconstructed, height_reconstructed))
            # 调整slice_2的尺寸
            resized_slice_2 = cv2.resize(slice_22, (width_reconstructed, height_reconstructed))
            print("resized_slice_1_shape:", resized_slice_1.shape)
            print("resized_slice_2_shape:",  resized_slice_2.shape)
            print("reconstructed_data_shape:", reconstructed_data[:, :, i-1].shape)
            # 将切片展平为一维数组
            resized_slice_1_flat = resized_slice_1.flatten()
            resized_slice_2_flat = resized_slice_2.flatten()
            reconstructed_data_flat = reconstructed_data[:, :, i-1].flatten()

            print("resized_slice_1_flat shape:", resized_slice_1_flat.shape)
            print("resized_slice_2_flat shape:", resized_slice_2_flat.shape)
            print("reconstructed_data_flat shape:", reconstructed_data_flat.shape)
            if reconstructed_data_flat.shape[0] > resized_slice_1_flat.shape[0]:
                resized_slice_1_flat = np.pad(resized_slice_1_flat, (0, reconstructed_data_flat.shape[0] - resized_slice_1_flat.shape[0]), mode='constant')
            else:
                resized_slice_1_flat = resized_slice_1_flat[:reconstructed_data_flat.shape[0]]
            # 计算相关系数
            corr_1 = np.corrcoef(resized_slice_1_flat, reconstructed_data_flat)[0, 1]
            corr_2 = np.corrcoef(resized_slice_2_flat, reconstructed_data_flat)[0, 1]
            if corr_1 > corr_2:
                reconstructed_data[:, :, i] = slice_1
            else:
                reconstructed_data[:, :, i] = slice_2
        # 创建并保存重构后的数据到新的文件
        # 检查目录是否存在，如果不存在则创建它
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        # 设置输出文件路径和文件名
        output_file = os.path.join(output_path, file_name, 'mri_norm2.h5')
        # 获取目录路径
        output_dir = os.path.dirname(output_file)
        # 检查目录是否存在，如果不存在则创建它
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 创建并保存重构后的数据到新的文件
        with h5py.File(output_file, 'w') as h5f_reconstructed:
            h5f_reconstructed.create_dataset('image', data=reconstructed_data.squeeze())
