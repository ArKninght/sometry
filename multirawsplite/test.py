import numpy as np
import os
import glob

def split_raw_file(input_file, output_dir='./', dtype=np.float32, width=972, height=768, num_images=5):
    """
    将包含多张CT投影图像的raw文件分割成多个独立的raw文件
    
    参数:
        input_file: 输入的raw文件路径
        output_dir: 输出目录,默认为当前目录
        dtype: 数据类型,默认为float32
        width: 图像宽度
        height: 图像高度
        num_images: 图像帧数
    """
    # 计算单张图像的大小(像素数)
    pixels_per_image = width * height
    bytes_per_pixel = np.dtype(dtype).itemsize
    single_image_size = pixels_per_image * bytes_per_pixel
    total_size = single_image_size * num_images
    
    # 读取整个raw文件
    try:
        with open(input_file, 'rb') as f:
            data = f.read()
    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file}")
        return None
    
    # 检查文件大小是否匹配
    actual_size = len(data)
    
    if actual_size != total_size:
        # 尝试自动检测数据类型
        detected_bytes_per_pixel = actual_size // (pixels_per_image * num_images)
        if detected_bytes_per_pixel == 1:
            dtype = np.uint8
            bytes_per_pixel = 1
        elif detected_bytes_per_pixel == 2:
            dtype = np.uint16
            bytes_per_pixel = 2
        elif detected_bytes_per_pixel == 4:
            dtype = np.float32
            bytes_per_pixel = 4
    
    # 使用numpy读取数据
    raw_data = np.frombuffer(data, dtype=dtype)
    
    # 分割图像数据并返回
    images = []
    for i in range(num_images):
        start_idx = i * pixels_per_image
        end_idx = (i + 1) * pixels_per_image
        image_data = raw_data[start_idx:end_idx]
        images.append(image_data)
    
    return images


def process_folder(input_folder, output_dir='./split_output', dtype=np.float32, width=972, height=768, num_images=5):
    """
    批量处理文件夹中的多帧图像文件
    将所有文件的第1帧存到image_1文件夹,第2帧存到image_2文件夹,以此类推
    
    参数:
        input_folder: 输入文件夹路径
        output_dir: 输出目录
        dtype: 数据类型
        width: 图像宽度
        height: 图像高度
        num_images: 每个文件的帧数
    """
    # 获取文件夹中所有.raw文件
    raw_files = sorted(glob.glob(os.path.join(input_folder, "*.raw")))
    
    if len(raw_files) == 0:
        print(f"错误: 在 {input_folder} 中未找到.raw文件")
        return
    
    print(f"找到 {len(raw_files)} 个raw文件")
    print(f"开始处理...\n")
    
    # 为每一帧创建文件夹
    for i in range(num_images):
        frame_folder = os.path.join(output_dir, f"image_{i+1}")
        os.makedirs(frame_folder, exist_ok=True)
    
    # 处理每个文件
    for file_idx, input_file in enumerate(raw_files):
        print(f"处理文件 [{file_idx+1}/{len(raw_files)}]: {os.path.basename(input_file)}")
        
        # 分割文件
        images = split_raw_file(input_file, output_dir, dtype, width, height, num_images)
        
        if images is None:
            continue
        
        # 保存每一帧到对应的文件夹
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        
        for frame_idx, image_data in enumerate(images):
            frame_folder = os.path.join(output_dir, f"image_{frame_idx+1}")
            output_file = os.path.join(frame_folder, f"{base_name}_frame_{frame_idx+1}.raw")
            
            # 保存为raw文件
            image_data.tofile(output_file)
            
        print(f"  已保存 {len(images)} 帧图像\n")
    
    print(f"批量处理完成!")
    print(f"输出目录: {output_dir}")
    print(f"文件结构: 每个image_X文件夹包含所有文件的第X帧")


def verify_split_results(output_dir, width=972, height=768, dtype=np.float32, num_images=5):
    """
    验证分割结果
    """
    print("\n验证分割结果:")
    
    for i in range(num_images):
        frame_folder = os.path.join(output_dir, f"image_{i+1}")
        
        if not os.path.exists(frame_folder):
            print(f"✗ 文件夹 image_{i+1} 不存在")
            continue
        
        raw_files = glob.glob(os.path.join(frame_folder, "*.raw"))
        
        if len(raw_files) == 0:
            print(f"✗ image_{i+1}: 未找到文件")
            continue
        
        # 检查第一个文件
        file_path = raw_files[0]
        data = np.fromfile(file_path, dtype=dtype)
        expected_pixels = width * height
        
        if len(data) == expected_pixels:
            print(f"✓ image_{i+1}: {len(raw_files)} 个文件, 每个 {len(data)} 像素, 数据范围: {data.min():.2f} - {data.max():.2f}")
        else:
            print(f"✗ image_{i+1}: 大小不匹配 (预期 {expected_pixels}, 实际 {len(data)})")


if __name__ == "__main__":
    # 输入文件夹路径
    input_folder = "/mnt/d/多帧扫描/1.2.0.20251014.153553.mice2-10帧/1/1"
    
    # 输出目录
    output_dir = "/mnt/d/多帧扫描/1.2.0.20251014.153553.mice2-10帧/1/split_output"
    
    # 图像参数
    width = 972
    height = 768
    num_images = 10  # 每个文件包含10帧
    dtype = np.uint16  # 原始数据类型为uint16
    
    # 批量处理文件夹
    process_folder(input_folder, output_dir, dtype, width, height, num_images)
    
    # 验证结果
    verify_split_results(output_dir, width, height, dtype, num_images)