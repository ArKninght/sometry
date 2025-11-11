#%% Import statements
import tigre
import numpy as np
from tigre.utilities import sample_loader
from tigre.utilities import CTnoise
import tigre.algorithms as algs
import pydicom
import os
from pathlib import Path

#%% DICOM 文件读取模块
def read_dicom_files(folder_path, sort_by_position=True):
    """
    读取文件夹中的所有 DICOM 文件并返回 numpy 数组
    直接创建 numpy 数组而不使用中间列表，节省内存
    
    Parameters:
    folder_path: DICOM 文件所在的文件夹路径
    sort_by_position: 是否按图像位置排序 (默认: True)
    
    Returns:
    numpy.ndarray: 包含 DICOM 图像数据的 3D numpy 数组，形状为 (slices, height, width)
    dict: 包含 DICOM 元数据信息的字典
    """
    dicom_files = []
    metadata = {}
    
    # 获取文件夹中的所有 .dcm 文件
    folder_path = Path(folder_path)
    dcm_files = list(folder_path.glob('*.dcm'))
    
    if not dcm_files:
        print(f"在 {folder_path} 中未找到 .dcm 文件")
        return np.array([]), {}
    
    print(f"找到 {len(dcm_files)} 个 DICOM 文件")
    
    # 第一步：读取文件头信息并确定维度
    for file_path in dcm_files:
        try:
            ds = pydicom.dcmread(file_path)
            dicom_files.append((ds, file_path))
            
            # 提取基本元数据（只从第一个文件）
            if not metadata:
                metadata = {
                    'PatientID': getattr(ds, 'PatientID', 'Unknown'),
                    'StudyDescription': getattr(ds, 'StudyDescription', 'Unknown'),
                    'SeriesDescription': getattr(ds, 'SeriesDescription', 'Unknown'),
                    'ImageShape': (ds.Rows, ds.Columns),
                    'PixelSpacing': getattr(ds, 'PixelSpacing', [1.0, 1.0]),
                    'SliceThickness': getattr(ds, 'SliceThickness', 1.0),
                    'NumberOfFiles': len(dcm_files)
                }
            
        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {e}")
            continue
    
    if not dicom_files:
        return np.array([]), metadata
    
    # 按图像位置排序（如果启用）
    if sort_by_position and dicom_files:
        try:
            dicom_files.sort(key=lambda x: float(getattr(x[0], 'ImagePositionPatient', [0, 0, 0])[2]))
            print("已按图像位置对 DICOM 文件进行排序")
        except:
            print("无法按位置排序，使用文件名排序")
            dicom_files.sort(key=lambda x: str(x[1]))
    
    # 第二步：获取第一个图像的维度信息，直接创建3D numpy数组
    first_ds = dicom_files[0][0]
    height, width = first_ds.Rows, first_ds.Columns
    num_slices = len(dicom_files)
    
    # 直接创建目标形状的numpy数组
    volume = np.zeros((num_slices, height, width), dtype=np.float32)
    print(f"预分配 3D numpy 数组，形状: {volume.shape}")
    
    # 第三步：直接将像素数据写入numpy数组的对应位置
    successful_reads = 0
    for i, (ds, file_path) in enumerate(dicom_files):
        try:
            pixel_array = ds.pixel_array.astype(np.float32)
            volume[i] = pixel_array
            successful_reads += 1
            
            # 释放DICOM数据对象以节省内存
            del ds
            
        except Exception as e:
            print(f"提取像素数据时出错 {file_path}: {e}")
            continue
    
    metadata['NumberOfFiles'] = successful_reads
    
    if successful_reads > 0:
        print(f"成功读取 {successful_reads} 个 DICOM 图像")
        print(f"3D 体积形状: {volume.shape}")
        print(f"数据类型: {volume.dtype}")
        print(f"内存使用估计: {volume.nbytes / 1024 / 1024:.2f} MB")
        
        # 如果有失败的读取，截取有效部分
        if successful_reads < num_slices:
            volume = volume[:successful_reads]
            print(f"截取有效数据，最终形状: {volume.shape}")
        
        return volume, metadata
    
    return np.array([]), metadata

def read_single_dicom(file_path):
    """
    读取单个 DICOM 文件
    
    Parameters:
    file_path: DICOM 文件路径
    
    Returns:
    numpy.ndarray: 图像数据
    dict: 基本元数据
    """
    try:
        ds = pydicom.dcmread(file_path)
        pixel_array = ds.pixel_array.astype(np.float32)
        
        metadata = {
            'PatientID': getattr(ds, 'PatientID', 'Unknown'),
            'StudyDescription': getattr(ds, 'StudyDescription', 'Unknown'),
            'SeriesDescription': getattr(ds, 'SeriesDescription', 'Unknown'),
            'ImageShape': (ds.Rows, ds.Columns),
            'PixelSpacing': getattr(ds, 'PixelSpacing', [1.0, 1.0]),
            'SliceThickness': getattr(ds, 'SliceThickness', 1.0)
        }
        
        print(f"成功读取 DICOM 文件: {file_path}")
        print(f"图像尺寸: {pixel_array.shape}")
        
        return pixel_array, metadata
        
    except Exception as e:
        print(f"读取 DICOM 文件失败 {file_path}: {e}")
        return None, None

# DICOM 使用示例
"""
使用示例：

# 1. 读取文件夹中的所有 DICOM 文件
dicom_volume, metadata = read_dicom_files('./dicom_folder/')

# 2. 读取单个 DICOM 文件
single_image, single_metadata = read_single_dicom('./example.dcm')

# 3. 使用返回的 numpy 数组
if dicom_volume.size > 0:
    print(f"DICOM 体积形状: {dicom_volume.shape}")
    first_slice = dicom_volume[0]  # 第一个切片
    middle_slice = dicom_volume[len(dicom_volume)//2]  # 中间切片
"""

#%% TIGRE CT 重建模块与 GPU 显存优化

def optimize_geometry_for_gpu(original_geo, reduction_factor=2):
    """
    优化几何参数以减少 GPU 显存使用
    
    Parameters:
    original_geo: 原始几何对象
    reduction_factor: 缩减因子，用于降低分辨率
    
    Returns:
    优化后的几何对象
    """
    geo_optimized = tigre.geometry()
    
    # 保持距离参数
    geo_optimized.DSD = original_geo.DSD
    geo_optimized.DSO = original_geo.DSO
    geo_optimized.COR = original_geo.COR
    geo_optimized.rotDetector = original_geo.rotDetector
    geo_optimized.mode = original_geo.mode
    geo_optimized.accuracy = original_geo.accuracy
    
    # 缩减探测器分辨率
    geo_optimized.nDetector = original_geo.nDetector // reduction_factor
    geo_optimized.dDetector = original_geo.dDetector * reduction_factor
    geo_optimized.sDetector = geo_optimized.nDetector * geo_optimized.dDetector
    
    # 缩减体素分辨率
    geo_optimized.nVoxel = original_geo.nVoxel // reduction_factor
    geo_optimized.sVoxel = original_geo.sVoxel  # 保持物理尺寸
    geo_optimized.dVoxel = geo_optimized.sVoxel / geo_optimized.nVoxel
    
    # 保持偏移
    geo_optimized.offOrigin = original_geo.offOrigin
    geo_optimized.offDetector = original_geo.offDetector
    
    print(f"优化前 - 探测器: {original_geo.nDetector}, 体素: {original_geo.nVoxel}")
    print(f"优化后 - 探测器: {geo_optimized.nDetector}, 体素: {geo_optimized.nVoxel}")
    
    return geo_optimized

def batch_reconstruct_fdk(projections, geo, angles, batch_size=50):
    """
    分批进行 FDK 重建以减少显存占用
    
    Parameters:
    projections: 投影数据
    geo: 几何对象
    angles: 投影角度
    batch_size: 每批处理的角度数量
    
    Returns:
    重建图像
    """
    num_angles = len(angles)
    print(f"总角度数: {num_angles}, 批处理大小: {batch_size}")
    
    # 初始化结果数组
    result_image = np.zeros(geo.nVoxel, dtype=np.float32)
    
    # 分批处理
    for i in range(0, num_angles, batch_size):
        end_idx = min(i + batch_size, num_angles)
        batch_angles = angles[i:end_idx]
        batch_projections = projections[i:end_idx]
        
        print(f"处理批次: {i//batch_size + 1}/{(num_angles-1)//batch_size + 1}, 角度 {i} 到 {end_idx-1}")

        batch_projections = log_normalize_projections(batch_projections, max_pixel_value=16383)

        try:
            # FDK 重建当前批次
            batch_result = algs.fdk(batch_projections, geo, batch_angles)
            # 按照角度权重累加结果
            result_image += batch_result * (len(batch_angles) / num_angles)
            
            # 释放 GPU 内存
            del batch_result
            import gc
            gc.collect()
            
        except Exception as e:
            print(f"批次 {i//batch_size + 1} 重建失败: {e}")
            continue
    
    return result_image

def downsample_projections(projections, factor=2):
    """
    降采样投影数据以减少内存占用
    
    Parameters:
    projections: 原始投影数据 (angles, height, width)
    factor: 降采样因子
    
    Returns:
    降采样后的投影数据
    """
    from scipy import ndimage
    
    print(f"原始投影形状: {projections.shape}")
    
    # 对每个投影进行降采样
    downsampled = np.zeros((projections.shape[0],
                           projections.shape[1]//factor,
                           projections.shape[2]//factor),
                          dtype=np.float32)
    
    for i in range(projections.shape[0]):
        downsampled[i] = ndimage.zoom(projections[i], 1/factor, order=1)
    
    print(f"降采样后投影形状: {downsampled.shape}")
    return downsampled

#%% 投影数据预处理模块

def log_normalize_projections(projections, max_pixel_value=16383, epsilon=1e-6):
    """
    对投影数据进行对数归一化处理，解决重建图像的色彩/对比度问题
    
    Parameters:
    projections: 投影数据 (numpy array)
    max_pixel_value: 单个像素的最大值 (默认: 16383 for 14-bit)
    epsilon: 防止对数计算中的零值 (默认: 1e-6)
    
    Returns:
    处理后的投影数据
    """
    print(f"投影预处理开始...")
    print(f"原始数据范围: [{projections.min():.4f}, {projections.max():.4f}]")
    print(f"数据类型: {projections.dtype}")
    
    # 确保数据为正值，避免对数计算问题
    projections_positive = np.maximum(projections, epsilon)
    
    # 对数变换：I = log(I0/I) 其中 I0 是入射强度，I 是透射强度
    # 这里假设 max_pixel_value 是入射强度 I0
    log_projections = np.log(max_pixel_value / projections_positive)
    
    # 处理可能的无穷大值（当像素值为0时）
    log_projections = np.where(np.isfinite(log_projections), log_projections, 0)
    
    print(f"对数变换后数据范围: [{log_projections.min():.4f}, {log_projections.max():.4f}]")
    
    return log_projections.astype(np.float32)

def advanced_normalize_projections(projections, max_pixel_value=16383,
                                 normalization_method='log',
                                 outlier_percentile=99.5):
    """
    高级投影数据预处理，包含多种归一化方法
    
    Parameters:
    projections: 投影数据
    max_pixel_value: 最大像素值
    normalization_method: 归一化方法 ('log', 'linear', 'robust')
    outlier_percentile: 异常值处理百分位数
    
    Returns:
    处理后的投影数据
    """
    print(f"高级投影预处理开始，方法: {normalization_method}")
    print(f"原始数据统计:")
    print(f"  - 形状: {projections.shape}")
    print(f"  - 范围: [{projections.min():.2f}, {projections.max():.2f}]")
    print(f"  - 均值: {projections.mean():.2f}")
    print(f"  - 标准差: {projections.std():.2f}")
    
    # 复制数据避免修改原始数据
    processed = projections.copy().astype(np.float32)
    
    # 异常值处理
    upper_threshold = np.percentile(processed, outlier_percentile)
    processed = np.clip(processed, 0, upper_threshold)
    
    if normalization_method == 'log':
        # 对数归一化 (Beer-Lambert 定律)
        epsilon = processed.max() * 1e-6
        processed = np.maximum(processed, epsilon)
        processed = np.log(max_pixel_value / processed)
        processed = np.where(np.isfinite(processed), processed, 0)
        
    elif normalization_method == 'linear':
        # 线性归一化到 [0, 1]
        processed = (processed - processed.min()) / (processed.max() - processed.min())
        
    elif normalization_method == 'robust':
        # 鲁棒归一化（使用中位数和MAD）
        median = np.median(processed)
        mad = np.median(np.abs(processed - median))
        processed = (processed - median) / (mad + 1e-6)
    
    print(f"处理后数据统计:")
    print(f"  - 范围: [{processed.min():.4f}, {processed.max():.4f}]")
    print(f"  - 均值: {processed.mean():.4f}")
    print(f"  - 标准差: {processed.std():.4f}")
    
    return processed

def flat_field_correction(projections, flat_field=None, dark_field=None):
    """
    平场校正和暗场校正
    
    Parameters:
    projections: 原始投影数据
    flat_field: 平场图像 (空气中的投影)
    dark_field: 暗场图像 (无X射线时的投影)
    
    Returns:
    校正后的投影数据
    """
    if flat_field is None and dark_field is None:
        print("未提供平场或暗场数据，跳过校正")
        return projections
    
    corrected = projections.copy().astype(np.float32)
    
    if dark_field is not None:
        print("应用暗场校正...")
        corrected = corrected - dark_field
        corrected = np.maximum(corrected, 0)  # 确保非负值
    
    if flat_field is not None:
        print("应用平场校正...")
        if dark_field is not None:
            flat_corrected = flat_field - dark_field
        else:
            flat_corrected = flat_field
        
        # 避免除零
        flat_corrected = np.maximum(flat_corrected, flat_corrected.max() * 1e-6)
        corrected = corrected / flat_corrected
        
        # 去除异常值
        corrected = np.where(np.isfinite(corrected), corrected, 0)
    
    return corrected

# 创建几何对象
geo = tigre.geometry()
# Distances
geo.DSD = 1566.30  # Distance Source Detector      (mm)
geo.DSO = 1566.30  # Distance Source Origin        (mm)
# Detector parameters
geo.nDetector = np.array([768, 1032])  # number of pixels              (px)
geo.dDetector = np.array([1.0, 1.0])  # size of each pixel            (mm)
geo.sDetector = geo.nDetector * geo.dDetector  # total size of the detector    (mm)
# Image parameters
geo.nVoxel = np.array([50, 815, 815])  # number of voxels              (vx)
geo.dVoxel = np.array([1.0, 1.0, 1.0])  # size of each voxel (mm) - 按要求设置
geo.sVoxel = geo.nVoxel * geo.dVoxel  # total size of the image (mm) - 重新计算
# # Offsets # 516.0080 367.32
geo.offOrigin = np.array([0, 367.32-768.0/2, 516.0080-1032.0/2])  # Offset of image from origin   (mm)
geo.offDetector = np.array([367.32-768.0/2, 516.0080-1032.0/2])  # Offset of Detector            (mm)
# NoOffsets
# geo.offOrigin = np.array([0, 0, 0])  # No Offset of image from origin   (mm)
# geo.offDetector = np.array([0, 0])  # No Offset of Detector            (mm)
geo.accuracy = 0.5  # Variable to define accuracy of

geo.COR = 0  # y direction displacement for

geo.rotDetector = np.array([0, 0, 0])  # Rotation of the detector, by
geo.mode = "cone"  # Or 'parallel'. Geometry type.

# 定义投影角度
angles = np.linspace(0, 2 * np.pi, 720)  # 720个角度，从0到2π
print(f"投影角度数量: {len(angles)}")

# 读取DICOM文件夹，即为矫正后的投影文件
sino_data, sino_metadata = read_dicom_files('/mnt/d/0916/1.2.0.20250916.170544.mice1-720-2/1/calibrate/')
if sino_data.size == 0:
    raise ValueError("未能读取任何DICOM投影数据，请检查文件路径和内容。")

print(f"投影数据内存占用: {sino_data.nbytes / 1024 / 1024:.2f} MB")

# GPU 显存优化策略
print("\n开始 FDK 重建 - 使用 GPU 显存优化...")

# try:
#     # 策略1: 直接重建（如果显存足够）
#     print("策略1: 尝试直接 FDK 重建...")
#     imgFDK = algs.fdk(sino_data, geo, angles)
#     print(f"直接重建成功! 图像形状: {imgFDK.shape}")
    
# except Exception as e:
# print(f"直接重建失败: {e}")
print("尝试 GPU 显存优化策略...")

try:
    # 策略2: 分批重建
    print("策略2: 尝试分批重建...")
    imgFDK = batch_reconstruct_fdk(sino_data, geo, angles, batch_size=90)
    print(f"分批重建成功! 图像形状: {imgFDK.shape}")
    
except Exception as e:
    print(f"分批重建失败: {e}")
    
    try:
        # 策略3: 降采样投影数据
        print("策略3: 尝试降采样投影数据...")
        downsampled_sino = downsample_projections(sino_data, factor=2)
        
        # 对应调整几何参数
        geo_down = optimize_geometry_for_gpu(geo, reduction_factor=2)
        
        imgFDK = algs.fdk(downsampled_sino, geo_down, angles)
        print(f"降采样重建成功! 图像形状: {imgFDK.shape}")
        
    except Exception as e:
        print(f"降采样重建失败: {e}")
        
        try:
            # 策略4: 减少角度 + 分批处理
            print("策略4: 尝试减少角度并分批处理...")
            reduced_angles = angles[::2]  # 每两个角度取一个
            reduced_sino = sino_data[::2]
            
            imgFDK = batch_reconstruct_fdk(reduced_sino, geo, reduced_angles, batch_size=30)
            print(f"减少角度分批重建成功! 图像形状: {imgFDK.shape}")
            
        except Exception as e:
            print(f"所有 GPU 显存优化策略都失败了: {e}")
            print("建议: 1) 增加 GPU 显存; 2) 进一步降低分辨率; 3) 使用 CPU 算法")
            imgFDK = np.zeros(geo.nVoxel, dtype=np.float32)

# 保存重建图像为 .raw 文件
def save_array_to_raw(array, filename):
    """
    将 numpy 数组保存为 raw 文件
    
    Parameters:
    array: numpy 数组
    filename: 输出文件名
    """
    array.astype(np.float32).tofile(filename)
    print(f"图像已保存到: {filename}")
    print(f"保存的图像形状: {array.shape}")
    print(f"数据类型: {array.dtype}")

# 保存重建结果
save_array_to_raw(imgFDK, 'reconstructed_image.raw')
