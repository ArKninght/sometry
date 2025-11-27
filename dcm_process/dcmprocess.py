#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DICOM批量处理工具
读取指定目录的DICOM文件,组合成3D体块并保存为RAW格式
支持多种输出数据类型
"""

import argparse
import glob
import os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pydicom


class DicomToRawConverter:
    """DICOM到RAW格式转换器"""
    
    def __init__(self, dicom_folder: str, output_folder: str = "./raw_output"):
        """
        初始化转换器
        
        Args:
            dicom_folder: DICOM文件所在文件夹
            output_folder: 输出RAW文件的文件夹
        """
        self.dicom_folder = dicom_folder
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)
        
    def collect_dicom_files(self) -> List[str]:
        """
        收集并排序DICOM文件
        
        Returns:
            排序后的DICOM文件路径列表
        """
        print(f"扫描DICOM文件: {self.dicom_folder}")
        
        # 查找所有.dcm文件
        dicom_files = glob.glob(os.path.join(self.dicom_folder, "*.dcm"))
        
        if not dicom_files:
            # 尝试不带扩展名的文件
            all_files = glob.glob(os.path.join(self.dicom_folder, "*"))
            dicom_files = []
            for f in all_files:
                if os.path.isfile(f):
                    try:
                        pydicom.dcmread(f, stop_before_pixels=True)
                        dicom_files.append(f)
                    except:
                        continue
        
        if not dicom_files:
            raise ValueError(f"在 {self.dicom_folder} 中未找到DICOM文件")
        
        print(f"  找到 {len(dicom_files)} 个DICOM文件")
        
        # 读取并按位置排序
        dicom_data = []
        for dcm_file in dicom_files:
            try:
                dcm = pydicom.dcmread(dcm_file, stop_before_pixels=True)
                # 获取切片位置
                if hasattr(dcm, 'ImagePositionPatient'):
                    z_pos = float(dcm.ImagePositionPatient[2])
                elif hasattr(dcm, 'SliceLocation'):
                    z_pos = float(dcm.SliceLocation)
                elif hasattr(dcm, 'InstanceNumber'):
                    z_pos = float(dcm.InstanceNumber)
                else:
                    z_pos = 0
                
                dicom_data.append((z_pos, dcm_file))
            except Exception as e:
                print(f"  警告: 无法读取 {dcm_file}: {e}")
                continue
        
        # 按z位置排序
        dicom_data.sort(key=lambda x: x[0])
        sorted_files = [f for _, f in dicom_data]
        
        print(f"  成功排序 {len(sorted_files)} 个DICOM文件")
        
        return sorted_files
    
    def read_dicom_series(self, dicom_files: List[str]) -> Tuple[np.ndarray, dict]:
        """
        读取DICOM序列
        
        Args:
            dicom_files: DICOM文件路径列表
            
        Returns:
            (3D numpy数组, 元数据字典)
        """
        print("\n读取DICOM序列...")
        
        # 读取第一个文件获取尺寸信息
        first_dcm = pydicom.dcmread(dicom_files[0])
        rows = int(first_dcm.Rows)
        cols = int(first_dcm.Columns)
        num_slices = len(dicom_files)
        
        print(f"  图像尺寸: {cols} × {rows} × {num_slices} (X×Y×Z)")
        
        # 初始化3D数组
        volume = np.zeros((num_slices, rows, cols), dtype=np.float32)
        
        # 提取元数据
        metadata = {
            'pixel_spacing': [1.0, 1.0],
            'slice_thickness': 1.0
        }
        
        if hasattr(first_dcm, 'PixelSpacing'):
            metadata['pixel_spacing'] = [float(x) for x in first_dcm.PixelSpacing]
        
        if hasattr(first_dcm, 'SliceThickness'):
            metadata['slice_thickness'] = float(first_dcm.SliceThickness)
        
        # 读取所有切片
        for i, dcm_file in enumerate(dicom_files):
            try:
                dcm = pydicom.dcmread(dcm_file)
                pixel_array = dcm.pixel_array.astype(np.float32)
                
                # 应用Rescale Slope和Intercept (将像素值转换为HU值)
                if hasattr(dcm, 'RescaleSlope') and hasattr(dcm, 'RescaleIntercept'):
                    slope = float(dcm.RescaleSlope)
                    intercept = float(dcm.RescaleIntercept)
                    pixel_array = pixel_array * slope + intercept
                
                volume[i, :, :] = pixel_array
                
                if (i + 1) % 50 == 0:
                    print(f"  已读取 {i + 1}/{num_slices} 个切片")
                    
            except Exception as e:
                print(f"  警告: 读取切片 {i} 失败: {e}")
                continue
        
        print(f"  完成读取")
        print(f"  数据范围: [{volume.min():.2f}, {volume.max():.2f}] HU")
        print(f"  体素尺寸: {metadata['pixel_spacing'][1]:.3f} × {metadata['pixel_spacing'][0]:.3f} × {metadata['slice_thickness']:.3f} mm³")
        
        return volume, metadata
    
    def convert_dtype(self, volume: np.ndarray, target_dtype: str) -> np.ndarray:
        """
        转换数据类型
        
        Args:
            volume: 输入3D数组
            target_dtype: 目标数据类型
        
        Returns:
            转换后的数组
        """
        dtype_map = {
            'float32': np.float32,
            'float64': np.float64,
            'int16': np.int16,
            'uint16': np.uint16,
            'uint8': np.uint8,
            'int32': np.int32,
            'uint32': np.uint32,
        }
        
        if target_dtype not in dtype_map:
            raise ValueError(f"不支持的数据类型: {target_dtype}. 支持: {list(dtype_map.keys())}")
        
        print(f"\n转换数据类型: {volume.dtype} -> {target_dtype}")
        print(f"  原始范围: [{volume.min():.2f}, {volume.max():.2f}]")
        
        np_dtype = dtype_map[target_dtype]
        
        # 对于整数类型,需要裁剪到合理范围
        if np.issubdtype(np_dtype, np.integer):
            dtype_info = np.iinfo(np_dtype)
            volume_clipped = np.clip(volume, dtype_info.min, dtype_info.max)
            converted = volume_clipped.astype(np_dtype)
            print(f"  裁剪范围: [{dtype_info.min}, {dtype_info.max}]")
        else:
            converted = volume.astype(np_dtype)
        
        print(f"  转换后范围: [{converted.min():.2f}, {converted.max():.2f}]")
        
        return converted
    
    def save_as_raw(self, volume: np.ndarray, output_filename: str, 
                    metadata: dict, order: str = 'C') -> str:
        """
        保存为RAW文件
        
        Args:
            volume: 3D numpy数组 (Z, Y, X)
            output_filename: 输出文件名(不含扩展名)
            metadata: 元数据字典
            order: 存储顺序 ('C'行优先, 'F'列优先)
        
        Returns:
            输出文件路径
        """
        print(f"\n保存为RAW文件...")
        
        # 生成输出路径
        dtype_suffix = str(volume.dtype)
        output_path = os.path.join(self.output_folder, f"{output_filename}_{dtype_suffix}.raw")
        
        print(f"  输出路径: {output_path}")
        print(f"  数据类型: {volume.dtype}")
        print(f"  形状: {volume.shape} (Z, Y, X)")
        print(f"  存储顺序: {'行优先(C)' if order == 'C' else '列优先(F)'}")
        
        # 保存为RAW文件
        volume.tofile(output_path, sep='', format=order)
        
        file_size = os.path.getsize(output_path) / (1024**2)
        print(f"  文件大小: {file_size:.2f} MB")
        
        # 保存元数据信息
        meta_path = output_path.replace('.raw', '_info.txt')
        with open(meta_path, 'w', encoding='utf-8') as f:
            f.write(f"RAW文件信息\n")
            f.write(f"{'='*60}\n")
            f.write(f"源DICOM目录: {self.dicom_folder}\n")
            f.write(f"\n")
            f.write(f"数据形状: {volume.shape} (Z, Y, X)\n")
            f.write(f"数据类型: {volume.dtype}\n")
            f.write(f"存储顺序: {'行优先(C)' if order == 'C' else '列优先(F)'}\n")
            f.write(f"数值范围: [{volume.min():.2f}, {volume.max():.2f}]\n")
            f.write(f"文件大小: {file_size:.2f} MB\n")
            f.write(f"\n")
            f.write(f"体素尺寸:\n")
            f.write(f"  X: {metadata['pixel_spacing'][1]:.3f} mm\n")
            f.write(f"  Y: {metadata['pixel_spacing'][0]:.3f} mm\n")
            f.write(f"  Z: {metadata['slice_thickness']:.3f} mm\n")
            f.write(f"\n")
            f.write(f"读取说明:\n")
            f.write(f"  Python: np.fromfile('{os.path.basename(output_path)}', dtype='{volume.dtype}').reshape({volume.shape})\n")
        
        print(f"  ✓ 元数据已保存: {meta_path}")
        print(f"  ✓ RAW文件保存完成")
        
        return output_path
    
    def process(self, output_filename: str = "volume", 
                target_dtype: str = 'float32',
                apply_flip_y: bool = True,
                storage_order: str = 'C') -> str:
        """
        执行完整的转换流程
        
        Args:
            output_filename: 输出文件名(不含扩展名和数据类型后缀)
            target_dtype: 目标数据类型
            apply_flip_y: 是否执行Y轴翻转
            storage_order: 存储顺序 ('C' or 'F')
        
        Returns:
            输出RAW文件路径
        """
        print("="*60)
        print("DICOM到RAW批量转换工具")
        print("="*60)
        
        try:
            # 1. 收集DICOM文件
            dicom_files = self.collect_dicom_files()
            
            # 2. 读取DICOM序列
            volume, metadata = self.read_dicom_series(dicom_files)
            
            # 3. Y轴翻转(可选)
            if apply_flip_y:
                print(f"\n执行Y轴翻转...")
                print(f"  翻转前形状: {volume.shape}")
                volume = np.flip(volume, axis=1)  # axis=1 对应 y 轴
                print(f"  翻转后形状: {volume.shape}")
                print(f"  ✓ Y轴翻转完成")
            
            # 4. 转换数据类型
            volume = self.convert_dtype(volume, target_dtype)
            
            # 5. 保存为RAW文件
            output_path = self.save_as_raw(volume, output_filename, metadata, storage_order)
            
            print("\n" + "="*60)
            print("转换完成!")
            print("="*60)
            
            return output_path
            
        except Exception as e:
            print(f"\n错误: {e}")
            import traceback
            traceback.print_exc()
            raise


def parse_args():
    parser = argparse.ArgumentParser(
        description="DICOM批量处理工具 - 将DICOM序列转换为RAW体块数据",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="DICOM文件所在文件夹路径"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="./raw_output",
        help="输出RAW文件的文件夹路径"
    )
    parser.add_argument(
        "-n", "--name",
        default="volume",
        help="输出文件名(不含扩展名)"
    )
    parser.add_argument(
        "--dtype",
        choices=['float32', 'float64', 'int16', 'uint16', 'uint8', 'int32', 'uint32'],
        default='float32',
        help="输出数据类型"
    )
    parser.add_argument(
        "--no-flip-y",
        action="store_true",
        help="不执行Y轴翻转"
    )
    parser.add_argument(
        "--order",
        choices=['C', 'F'],
        default='C',
        help="存储顺序: C(行优先), F(列优先)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 检查输入目录
    if not os.path.exists(args.input):
        print(f"错误: 输入目录不存在: {args.input}")
        return
    
    # 创建转换器并执行
    converter = DicomToRawConverter(
        dicom_folder=args.input,
        output_folder=args.output_dir
    )
    
    converter.process(
        output_filename=args.name,
        target_dtype=args.dtype,
        apply_flip_y=not args.no_flip_y,
        storage_order=args.order
    )


if __name__ == "__main__":
    main()