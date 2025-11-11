import pydicom
import numpy as np
import os
import glob
from pathlib import Path


def crop_dcm_to_square(input_path, output_path, target_size=767):
    """
    将DCM文件裁剪成正方形
    
    参数:
        input_path: 输入DCM文件路径
        output_path: 输出DCM文件路径
        target_size: 目标尺寸（默认767*767）
    
    返回:
        True表示成功，False表示失败
    """
    try:
        # 读取DCM文件
        ds = pydicom.dcmread(input_path)
        
        # 获取原始图像数据
        pixel_array = ds.pixel_array
        
        # 获取原始图像尺寸
        original_height, original_width = pixel_array.shape
        
        print(f"  原始尺寸: {original_width} x {original_height}")
        
        # 计算裁剪区域（从中心裁剪）
        if original_width >= target_size and original_height >= target_size:
            # 计算起始坐标（中心裁剪）
            start_x = (original_width - target_size) // 2
            start_y = (original_height - target_size) // 2
            
            # 确保坐标为整数
            start_x = int(start_x)
            start_y = int(start_y)
            
            # 计算结束坐标
            end_x = start_x + target_size
            end_y = start_y + target_size
            
            # 裁剪图像
            cropped_array = pixel_array[start_y:end_y, start_x:end_x]
            
            print(f"  裁剪区域: ({start_x}, {start_y}) 到 ({end_x}, {end_y})")
            print(f"  裁剪后尺寸: {target_size} x {target_size}")
            
            # 更新像素数据
            ds.PixelData = cropped_array.tobytes()
            
            # 更新DICOM标签中的尺寸信息
            ds.Rows = target_size
            ds.Columns = target_size
            
            # 添加处理信息
            if hasattr(ds, 'ImageComments'):
                ds.ImageComments = f"Cropped to {target_size}x{target_size} from center"
            else:
                ds.add_new([0x0020, 0x4000], 'LT', f"Cropped to {target_size}x{target_size} from center")
            
            # 保存裁剪后的文件
            ds.save_as(output_path)
            
            print(f"  ✓ 成功保存到: {os.path.basename(output_path)}")
            return True
            
        else:
            print(f"  ✗ 图像尺寸小于目标尺寸 {target_size}x{target_size}")
            print(f"     原始尺寸: {original_width}x{original_height}")
            return False
            
    except Exception as e:
        print(f"  ✗ 处理失败: {e}")
        return False


def process_dcm_folder(input_folder, output_folder, target_size=767):
    """
    批量处理文件夹中的所有DCM文件
    
    参数:
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径
        target_size: 目标尺寸（默认767*767）
    """
    print(f"\n{'=' * 60}")
    print(f"DCM文件裁剪工具")
    print(f"{'=' * 60}")
    print(f"输入文件夹: {input_folder}")
    print(f"输出文件夹: {output_folder}")
    print(f"目标尺寸: {target_size} x {target_size}")
    print(f"{'=' * 60}\n")
    
    # 检查输入文件夹是否存在
    if not os.path.exists(input_folder):
        print(f"错误: 输入文件夹 {input_folder} 不存在")
        return
    
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取所有DCM文件
    dcm_files = glob.glob(os.path.join(input_folder, "*.dcm"))
    
    if not dcm_files:
        print(f"错误: 在 {input_folder} 中未找到DCM文件")
        return
    
    print(f"找到 {len(dcm_files)} 个DCM文件\n")
    
    # 处理每个文件
    success_count = 0
    failed_count = 0
    
    for i, input_file in enumerate(dcm_files, 1):
        file_name = os.path.basename(input_file)
        output_file = os.path.join(output_folder, file_name)
        
        print(f"[{i}/{len(dcm_files)}] 处理: {file_name}")
        
        # 裁剪文件
        if crop_dcm_to_square(input_file, output_file, target_size):
            success_count += 1
        else:
            failed_count += 1
        print()
    
    # 输出统计信息
    print(f"{'=' * 60}")
    print(f"处理完成!")
    print(f"  成功: {success_count} 个文件")
    print(f"  失败: {failed_count} 个文件")
    print(f"  输出目录: {output_folder}")
    print(f"{'=' * 60}\n")


def process_multiple_folders(folder_list, base_output_dir, target_size=767):
    """
    处理多个文件夹中的DCM文件
    
    参数:
        folder_list: 输入文件夹路径列表
        base_output_dir: 基础输出目录
        target_size: 目标尺寸（默认767*767）
    """
    print(f"\n{'=' * 60}")
    print(f"批量DCM文件裁剪工具")
    print(f"{'=' * 60}")
    print(f"目标尺寸: {target_size} x {target_size}")
    print(f"基础输出目录: {base_output_dir}")
    print(f"{'=' * 60}\n")
    
    total_success = 0
    total_failed = 0
    
    for i, input_folder in enumerate(folder_list, 1):
        if not os.path.exists(input_folder):
            print(f"[{i}/{len(folder_list)}] 跳过（文件夹不存在）: {input_folder}")
            continue
        
        # 为每个输入文件夹创建对应的输出文件夹
        folder_name = os.path.basename(input_folder.rstrip('/\\'))
        output_folder = os.path.join(base_output_dir, f"{folder_name}_cropped")
        
        print(f"\n[{i}/{len(folder_list)}] 处理文件夹: {folder_name}")
        
        # 获取DCM文件数量
        dcm_files = glob.glob(os.path.join(input_folder, "*.dcm"))
        if not dcm_files:
            print(f"  跳过（无DCM文件）")
            continue
        
        print(f"  包含 {len(dcm_files)} 个DCM文件")
        
        # 处理该文件夹
        success_count, failed_count = process_single_folder(input_folder, output_folder, target_size)
        
        total_success += success_count
        total_failed += failed_count
    
    # 输出总体统计
    print(f"\n{'#' * 60}")
    print(f"全部处理完成!")
    print(f"  总成功: {total_success} 个文件")
    print(f"  总失败: {total_failed} 个文件")
    print(f"{'#' * 60}\n")


def process_single_folder(input_folder, output_folder, target_size=767):
    """
    处理单个文件夹（辅助函数，不打印标题）
    
    返回:
        (success_count, failed_count)
    """
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取所有DCM文件
    dcm_files = glob.glob(os.path.join(input_folder, "*.dcm"))
    
    success_count = 0
    failed_count = 0
    
    for input_file in dcm_files:
        file_name = os.path.basename(input_file)
        output_file = os.path.join(output_folder, file_name)
        
        # 裁剪文件（静默模式）
        if crop_dcm_to_square(input_file, output_file, target_size):
            success_count += 1
        else:
            failed_count += 1
    
    return success_count, failed_count


if __name__ == "__main__":
    # 配置参数
    TARGET_SIZE = 767  # 目标尺寸
    
    # 选项1: 处理单个文件夹
    input_folder = "/mnt/d/多帧扫描/recon/recon1/noised"  # 修改为你的输入文件夹路径
    output_folder = "/mnt/d/多帧扫描/recon/recon1/noised_cropped"  # 修改为你的输出文件夹路径
    
    # 选项2: 处理多个文件夹（取消注释使用）
    # input_folders = [
    #     "./folder1",
    #     "./folder2", 
    #     "./folder3",
    # ]
    # base_output_dir = "./cropped_output"
    
    print("DCM文件裁剪工具")
    print("=" * 60)
    print(f"此工具将DCM文件裁剪成 {TARGET_SIZE}x{TARGET_SIZE} 大小")
    print("裁剪方式：从图像中心裁剪")
    print("=" * 60)
    
    # 检查输入文件夹是否存在
    if not os.path.exists(input_folder):
        print(f"\n错误: 输入文件夹 {input_folder} 不存在")
        print("请修改代码中的 input_folder 变量")
    else:
        print(f"\n配置信息:")
        print(f"输入文件夹: {input_folder}")
        print(f"输出文件夹: {output_folder}")
        print(f"目标尺寸: {TARGET_SIZE} x {TARGET_SIZE}")
        
        user_input = input("\n是否开始处理？(y/n): ").strip().lower()
        if user_input == 'y':
            # 处理单个文件夹
            process_dcm_folder(input_folder, output_folder, TARGET_SIZE)
        else:
            print("\n已取消操作")
    
    # 处理多个文件夹的示例（已注释）
    """
    print("\n批量处理模式")
    for folder in input_folders:
        exists = "✓" if os.path.exists(folder) else "✗"
        print(f"  {exists} {folder}")
    
    if all(os.path.exists(f) for f in input_folders):
        user_input = input("\n是否开始批量处理？(y/n): ").strip().lower()
        if user_input == 'y':
            process_multiple_folders(input_folders, base_output_dir, TARGET_SIZE)
        else:
            print("\n已取消操作")
    else:
        print("\n错误: 部分文件夹不存在")
    """