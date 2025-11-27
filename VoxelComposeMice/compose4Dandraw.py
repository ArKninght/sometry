#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将处理后的ROI体块插入到原始大小的体块中,并保存为RAW文件

功能:
1. 从DICOM文件夹读取原始大小的体块
2. 从NIfTI文件读取处理后的ROI体块
3. 从YAML文件读取ROI位置信息
4. 将ROI插入到原始体块的指定位置
5. 保存为RAW文件
"""

import argparse
import os
from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np
import yaml


class VolumeComposer:
    """体块组合器 - 将ROI插入到原始体块中"""
    
    def __init__(self, roi_nifti_path: str, dicom_folder: str, yaml_path: str, output_folder: str):
        """
        初始化组合器
        
        Args:
            roi_nifti_path: ROI体块NIfTI文件路径
            dicom_folder: 原始DICOM文件夹路径
            yaml_path: 包含ROI坐标信息的YAML文件路径
            output_folder: 输出RAW文件的文件夹路径
        """
        self.roi_nifti_path = roi_nifti_path
        self.dicom_folder = dicom_folder
        self.yaml_path = yaml_path
        self.output_folder = output_folder
        
        os.makedirs(output_folder, exist_ok=True)
        
        # 加载配置
        self.roi_coords = None
        self.original_shape = None
        self._load_yaml_config()
    
    def _load_yaml_config(self):
        """从YAML文件加载ROI坐标和原始形状信息"""
        print(f"读取配置文件: {self.yaml_path}")
        
        if not os.path.exists(self.yaml_path):
            raise FileNotFoundError(f"配置文件不存在: {self.yaml_path}")
        
        with open(self.yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if 'roi_coordinates' not in config:
            raise ValueError("YAML文件中缺少 'roi_coordinates' 字段")
        if 'original_volume_shape' not in config:
            raise ValueError("YAML文件中缺少 'original_volume_shape' 字段")
        
        # 提取ROI坐标
        self.roi_coords = {
            'x': (config['roi_coordinates']['x']['min'], config['roi_coordinates']['x']['max']),
            'y': (config['roi_coordinates']['y']['min'], config['roi_coordinates']['y']['max']),
            'z': (config['roi_coordinates']['z']['min'], config['roi_coordinates']['z']['max'])
        }
        
        # 提取原始形状
        self.original_shape = (
            config['original_volume_shape']['z'],
            config['original_volume_shape']['y'],
            config['original_volume_shape']['x']
        )
        
        print(f"  原始体块形状: {self.original_shape} (Z, Y, X)")
        print(f"  ROI位置:")
        print(f"    X: [{self.roi_coords['x'][0]}, {self.roi_coords['x'][1]})")
        print(f"    Y: [{self.roi_coords['y'][0]}, {self.roi_coords['y'][1]})")
        print(f"    Z: [{self.roi_coords['z'][0]}, {self.roi_coords['z'][1]})")
    
    def load_roi_nifti(self) -> np.ndarray:
        """加载ROI体块NIfTI文件"""
        print(f"\n加载ROI体块: {self.roi_nifti_path}")
        
        if not os.path.exists(self.roi_nifti_path):
            raise FileNotFoundError(f"ROI文件不存在: {self.roi_nifti_path}")
        
        img = nib.load(self.roi_nifti_path)
        # NIfTI存储格式是(X, Y, Z), 需要转置为(Z, Y, X)
        volume = img.get_fdata(dtype=np.float32)
        volume = np.transpose(volume, (2, 1, 0))
        
        print(f"  ROI形状: {volume.shape} (Z, Y, X)")
        print(f"  数据类型: {volume.dtype}")
        print(f"  数值范围: [{volume.min():.2f}, {volume.max():.2f}]")
        
        # 验证ROI大小与配置是否匹配
        expected_shape = (
            self.roi_coords['z'][1] - self.roi_coords['z'][0],
            self.roi_coords['y'][1] - self.roi_coords['y'][0],
            self.roi_coords['x'][1] - self.roi_coords['x'][0]
        )
        
        if volume.shape != expected_shape:
            print(f"  警告: ROI实际形状 {volume.shape} 与配置期望形状 {expected_shape} 不匹配")
            print(f"  将使用实际形状进行插入")
        
        return volume
    
    def load_original_volume_from_dicom(self) -> np.ndarray:
        """从DICOM文件夹读取原始体块"""
        print(f"\n从DICOM读取原始体块: {self.dicom_folder}")
        
        # 复用dcmcompose.py中的DicomToNiftiConverter
        from dcmcompose import DicomToNiftiConverter
        
        converter = DicomToNiftiConverter(self.dicom_folder, self.output_folder)
        
        # 收集并读取DICOM文件
        dicom_files = converter.collect_dicom_files()
        volume, metadata = converter.read_dicom_series(dicom_files)
        
        # Y轴翻转(与dcmcompose.py保持一致)
        print(f"  执行Y轴翻转...")
        volume = np.flip(volume, axis=1)
        
        print(f"  原始体块形状: {volume.shape} (Z, Y, X)")
        print(f"  数据类型: {volume.dtype}")
        print(f"  数值范围: [{volume.min():.2f}, {volume.max():.2f}] HU")
        
        # 验证形状是否与配置匹配
        if volume.shape != self.original_shape:
            print(f"  警告: 实际形状 {volume.shape} 与配置期望形状 {self.original_shape} 不匹配")
        
        return volume
    
    def insert_roi_into_volume(self, original_volume: np.ndarray, roi_volume: np.ndarray) -> np.ndarray:
        """将ROI插入到原始体块中"""
        print(f"\n将ROI插入到原始体块...")
        
        # 创建输出体块(复制原始数据)
        composed = original_volume.copy()
        
        # 获取ROI边界
        z_min, z_max = self.roi_coords['z']
        y_min, y_max = self.roi_coords['y']
        x_min, x_max = self.roi_coords['x']
        
        # 获取实际可插入的区域大小
        z_size = min(roi_volume.shape[0], z_max - z_min, composed.shape[0] - z_min)
        y_size = min(roi_volume.shape[1], y_max - y_min, composed.shape[1] - y_min)
        x_size = min(roi_volume.shape[2], x_max - x_min, composed.shape[2] - x_min)
        
        print(f"  插入位置: Z=[{z_min}, {z_min+z_size}), Y=[{y_min}, {y_min+y_size}), X=[{x_min}, {x_min+x_size})")
        print(f"  插入大小: {z_size} × {y_size} × {x_size}")
        
        # 执行插入
        composed[z_min:z_min+z_size, y_min:y_min+y_size, x_min:x_min+x_size] = \
            roi_volume[:z_size, :y_size, :x_size]
        
        print(f"  组合后形状: {composed.shape} (Z, Y, X)")
        print(f"  组合后数值范围: [{composed.min():.2f}, {composed.max():.2f}]")
        print(f"  ✓ ROI插入完成")
        
        return composed
    
    def save_as_raw(self, volume: np.ndarray, output_filename: str, dtype: str = 'float32') -> str:
        """
        保存为RAW文件
        
        Args:
            volume: 3D numpy数组 (Z, Y, X)
            output_filename: 输出文件名(不含扩展名)
            dtype: 输出数据类型
        
        Returns:
            输出文件路径
        """
        print(f"\n保存为RAW文件...")
        
        # 数据类型映射
        dtype_map = {
            'float32': np.float32,
            'float64': np.float64,
            'int16': np.int16,
            'uint16': np.uint16,
            'uint8': np.uint8,
            'int32': np.int32,
            'uint32': np.uint32,
        }
        
        if dtype not in dtype_map:
            raise ValueError(f"不支持的数据类型: {dtype}")
        
        # 转换数据类型
        output_volume = volume.astype(dtype_map[dtype])
        
        # 生成输出路径
        output_path = os.path.join(self.output_folder, f"{output_filename}_{dtype}.raw")
        
        print(f"  输出路径: {output_path}")
        print(f"  数据类型: {dtype}")
        print(f"  形状: {output_volume.shape} (Z, Y, X)")
        print(f"  数值范围: [{output_volume.min():.2f}, {output_volume.max():.2f}]")
        
        # 保存为RAW文件(行优先存储)
        output_volume.tofile(output_path)
        
        file_size = os.path.getsize(output_path) / (1024**2)
        print(f"  文件大小: {file_size:.2f} MB")
        
        # 保存元数据信息
        meta_path = output_path.replace('.raw', '_info.txt')
        with open(meta_path, 'w', encoding='utf-8') as f:
            f.write(f"RAW文件信息\n")
            f.write(f"{'='*60}\n")
            f.write(f"源ROI文件: {self.roi_nifti_path}\n")
            f.write(f"源DICOM目录: {self.dicom_folder}\n")
            f.write(f"配置文件: {self.yaml_path}\n")
            f.write(f"\n")
            f.write(f"数据形状: {output_volume.shape} (Z, Y, X)\n")
            f.write(f"数据类型: {dtype}\n")
            f.write(f"存储顺序: 行优先(C)\n")
            f.write(f"数值范围: [{output_volume.min():.2f}, {output_volume.max():.2f}]\n")
            f.write(f"文件大小: {file_size:.2f} MB\n")
            f.write(f"\n")
            f.write(f"ROI插入位置:\n")
            f.write(f"  X: [{self.roi_coords['x'][0]}, {self.roi_coords['x'][1]})\n")
            f.write(f"  Y: [{self.roi_coords['y'][0]}, {self.roi_coords['y'][1]})\n")
            f.write(f"  Z: [{self.roi_coords['z'][0]}, {self.roi_coords['z'][1]})\n")
        
        print(f"  ✓ 元数据已保存: {meta_path}")
        print(f"  ✓ RAW文件保存完成")
        
        return output_path
    
    def compose(self, output_filename: str = "composed_volume", dtype: str = 'float32') -> str:
        """
        执行完整的组合流程
        
        Args:
            output_filename: 输出文件名(不含扩展名和数据类型后缀)
            dtype: 输出数据类型
        
        Returns:
            输出RAW文件路径
        """
        print("="*60)
        print("体块组合程序 - 将ROI插入到原始体块")
        print("="*60)
        
        try:
            # 1. 加载ROI体块
            roi_volume = self.load_roi_nifti()
            
            # 2. 加载原始体块
            original_volume = self.load_original_volume_from_dicom()
            
            # 3. 插入ROI
            composed_volume = self.insert_roi_into_volume(original_volume, roi_volume)
            
            # 4. 保存为RAW文件
            output_path = self.save_as_raw(composed_volume, output_filename, dtype)
            
            print("\n" + "="*60)
            print("组合完成!")
            print("="*60)
            
            return output_path
            
        except Exception as e:
            print(f"\n错误: {e}")
            import traceback
            traceback.print_exc()
            raise


def parse_args():
    parser = argparse.ArgumentParser(
        description="将处理后的ROI体块插入到原始大小体块中并保存为RAW文件",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "-r", "--roi-nifti",
        required=True,
        default="/home/pby/Documents/someTry/VoxelComposeMice/4DCT_output_rawrange/phase_02_nearest.nii.gz",
        help="ROI体块NIfTI文件路径"
    )
    parser.add_argument(
        "-d", "--dicom-folder",
        default="/mnt/d/多帧扫描/recon/recon2/fusion/",
        help="原始DICOM文件夹路径"
    )
    parser.add_argument(
        "-y", "--yaml",
        default="/home/pby/Documents/someTry/VoxelComposeMice/nifti_output/hu_adjustment_stats.yaml",
        help="ROI坐标配置YAML文件路径"
    )
    parser.add_argument(
        "-o", "--output-folder",
        default="./composed_output",
        help="输出RAW文件的文件夹路径"
    )
    parser.add_argument(
        "-n", "--name",
        default="composed_volume",
        help="输出文件名(不含扩展名)"
    )
    parser.add_argument(
        "--dtype",
        choices=['float32', 'float64', 'int16', 'uint16', 'uint8', 'int32', 'uint32'],
        default='int16',
        help="RAW文件的数据类型"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 检查输入路径
    if not os.path.exists(args.roi_nifti):
        print(f"错误: ROI文件不存在: {args.roi_nifti}")
        return
    
    if not os.path.exists(args.dicom_folder):
        print(f"错误: DICOM文件夹不存在: {args.dicom_folder}")
        return
    
    if not os.path.exists(args.yaml):
        print(f"错误: YAML配置文件不存在: {args.yaml}")
        return
    
    # 创建组合器并执行
    composer = VolumeComposer(
        roi_nifti_path=args.roi_nifti,
        dicom_folder=args.dicom_folder,
        yaml_path=args.yaml,
        output_folder=args.output_folder
    )
    
    composer.compose(output_filename=args.name, dtype=args.dtype)


if __name__ == "__main__":
    main()