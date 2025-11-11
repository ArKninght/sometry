import pydicom
import numpy as np
import os
import glob
from pathlib import Path


def get_dcm_files_from_folders(folder_paths):
    """
    从多个文件夹中获取DCM文件，并按文件名分组
    
    参数:
        folder_paths: 文件夹路径列表
    
    返回:
        字典，key为文件名，value为该文件在各文件夹中的完整路径列表
    """
    dcm_groups = {}
    
    for folder in folder_paths:
        if not os.path.exists(folder):
            print(f"警告: 文件夹 {folder} 不存在，跳过")
            continue
        
        # 获取该文件夹中所有dcm文件
        dcm_files = glob.glob(os.path.join(folder, "*.dcm"))
        
        for dcm_path in dcm_files:
            # 获取文件名（不含路径）
            file_name = os.path.basename(dcm_path)
            
            # 将该文件路径添加到对应的分组中
            if file_name not in dcm_groups:
                dcm_groups[file_name] = []
            dcm_groups[file_name].append(dcm_path)
    
    return dcm_groups


def average_dcm_files(dcm_paths, output_path):
    """
    将多个DCM文件的像素数据相加并平均
    
    参数:
        dcm_paths: DCM文件路径列表
        output_path: 输出文件路径
    
    返回:
        True表示成功，False表示失败
    """
    if not dcm_paths:
        print("错误: 没有提供DCM文件路径")
        return False
    
    try:
        # 读取第一个DCM文件作为模板
        base_dcm = pydicom.dcmread(dcm_paths[0])
        
        # 获取像素数据
        pixel_arrays = []
        
        for dcm_path in dcm_paths:
            ds = pydicom.dcmread(dcm_path)
            pixel_array = ds.pixel_array.astype(np.float64)
            pixel_arrays.append(pixel_array)
        
        # 检查所有图像尺寸是否一致
        first_shape = pixel_arrays[0].shape
        for i, arr in enumerate(pixel_arrays[1:], 1):
            if arr.shape != first_shape:
                print(f"错误: 文件尺寸不一致")
                print(f"  文件1: {pixel_arrays[0].shape}")
                print(f"  文件{i+1}: {arr.shape}")
                return False
        
        # 计算平均值
        averaged_array = np.mean(pixel_arrays, axis=0)
        
        # 转换回原始数据类型
        original_dtype = base_dcm.pixel_array.dtype
        averaged_array = averaged_array.astype(original_dtype)
        
        # 更新像素数据
        base_dcm.PixelData = averaged_array.tobytes()
        
        # 添加处理信息到DICOM元数据
        if hasattr(base_dcm, 'ImageComments'):
            base_dcm.ImageComments = f"Averaged from {len(dcm_paths)} images"
        else:
            base_dcm.add_new([0x0020, 0x4000], 'LT', f"Averaged from {len(dcm_paths)} images")
        
        # 保存结果
        base_dcm.save_as(output_path)
        
        return True
        
    except Exception as e:
        print(f"错误: 处理DCM文件时出错: {e}")
        return False


def fusion_dcm_folders(folder_paths, output_dir, min_files=None):
    """
    融合多个文件夹中名称相同的DCM文件
    
    参数:
        folder_paths: 输入文件夹路径列表
        output_dir: 输出目录
        min_files: 最少需要的文件数量，如果为None则使用文件夹数量
    """
    # 如果未指定最少文件数，则默认要求所有文件夹都有该文件
    if min_files is None:
        min_files = len(folder_paths)
    
    print(f"\n{'=' * 60}")
    print(f"DCM文件融合程序")
    print(f"{'=' * 60}")
    print(f"输入文件夹数量: {len(folder_paths)}")
    print(f"输出目录: {output_dir}")
    print(f"最少文件数要求: {min_files}")
    print(f"{'=' * 60}\n")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有DCM文件并按文件名分组
    print("正在扫描文件夹...")
    dcm_groups = get_dcm_files_from_folders(folder_paths)
    
    if not dcm_groups:
        print("错误: 未找到任何DCM文件")
        return
    
    print(f"找到 {len(dcm_groups)} 个不同的文件名\n")
    
    # 统计信息
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    
    # 处理每组文件
    for file_name, paths in dcm_groups.items():
        print(f"处理: {file_name}")
        print(f"  找到 {len(paths)} 个文件")
        
        # 检查是否满足最少文件数要求
        if len(paths) < min_files:
            print(f"  ⚠ 跳过（文件数不足，需要至少 {min_files} 个）\n")
            skipped_count += 1
            continue
        
        # 显示文件来源
        for i, path in enumerate(paths, 1):
            folder_name = os.path.basename(os.path.dirname(path))
            print(f"    [{i}] {folder_name}")
        
        # 设置输出路径
        output_path = os.path.join(output_dir, file_name)
        
        # 执行融合
        print(f"  正在融合...")
        success = average_dcm_files(paths, output_path)
        
        if success:
            print(f"  ✓ 成功保存到: {os.path.basename(output_path)}\n")
            processed_count += 1
        else:
            print(f"  ✗ 处理失败\n")
            failed_count += 1
    
    # 输出统计信息
    print(f"{'=' * 60}")
    print(f"处理完成!")
    print(f"  成功处理: {processed_count} 个文件")
    print(f"  跳过: {skipped_count} 个文件")
    print(f"  失败: {failed_count} 个文件")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    # 配置输入文件夹路径（修改为你的实际路径）
    input_folders = [
        "/mnt/d/多帧扫描/1.2.0.20251014.153553.mice2-10帧/1/calibrate_1",
        "/mnt/d/多帧扫描/1.2.0.20251014.153553.mice2-10帧/1/calibrate_2",
        "/mnt/d/多帧扫描/1.2.0.20251014.153553.mice2-10帧/1/calibrate_3",
        "/mnt/d/多帧扫描/1.2.0.20251014.153553.mice2-10帧/1/calibrate_4",
        "/mnt/d/多帧扫描/1.2.0.20251014.153553.mice2-10帧/1/calibrate_5",
        "/mnt/d/多帧扫描/1.2.0.20251014.153553.mice2-10帧/1/calibrate_6",
        "/mnt/d/多帧扫描/1.2.0.20251014.153553.mice2-10帧/1/calibrate_7",
        "/mnt/d/多帧扫描/1.2.0.20251014.153553.mice2-10帧/1/calibrate_8",
        "/mnt/d/多帧扫描/1.2.0.20251014.153553.mice2-10帧/1/calibrate_9",
        "/mnt/d/多帧扫描/1.2.0.20251014.153553.mice2-10帧/1/calibrate_10",
    ]
    
    # 配置输出目录
    output_directory = "/mnt/d/多帧扫描/1.2.0.20251014.153553.mice2-10帧/1/calibrate_fusion"

    # 可选：设置最少文件数要求
    # 如果某个文件名在少于min_files个文件夹中存在，将被跳过
    # 设为None则要求所有文件夹都有该文件
    min_required_files = 5  # 或设为None表示要求全部5个文件夹都有
    
    print("DCM文件融合工具")
    print("=" * 60)
    print("此工具将多个文件夹中名称相同的DCM文件进行平均融合")
    print("=" * 60)
    print("\n配置信息:")
    print(f"输入文件夹:")
    for i, folder in enumerate(input_folders, 1):
        exists = "✓" if os.path.exists(folder) else "✗"
        print(f"  [{i}] {exists} {folder}")
    print(f"\n输出目录: {output_directory}")
    print(f"最少文件数: {min_required_files if min_required_files else '全部'}")
    
    # 检查是否至少有一个文件夹存在
    if not any(os.path.exists(f) for f in input_folders):
        print("\n错误: 没有找到任何存在的输入文件夹")
        print("请修改代码中的 input_folders 列表，设置正确的文件夹路径")
    else:
        user_input = input("\n是否开始处理？(y/n): ").strip().lower()
        if user_input == 'y':
            # 执行融合
            fusion_dcm_folders(input_folders, output_directory, min_required_files)
        else:
            print("\n已取消操作")