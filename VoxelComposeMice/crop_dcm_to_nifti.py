#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DICOM体块裁剪工具
根据YAML配置文件中的ROI坐标裁剪DICOM体块并保存为NIFTI格式
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import yaml
import pydicom
import nibabel as nib
from tqdm import tqdm


class DicomCropper:
    """DICOM体块裁剪器"""
    
    def __init__(self, dcm_folder: str, yaml_file: str, output_folder: str = "./cropped_nifti"):
        """
        初始化裁剪器
        
        Args:
            dcm_folder: DICOM文件夹路径
            yaml_file: YAML配置文件路径（包含ROI坐标信息）
            output_folder: 输出NIFTI文件夹
        """
        self.dcm_folder = dcm_folder
        self.yaml_file = yaml_file
        self.output_folder = output_folder
        
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"初始化DICOM裁剪器")
        print(f"  DICOM文件夹: {dcm_folder}")
        print(f"  YAML配置: {yaml_file}")
        print(f"  输出文件夹: {output_folder}")
    
    def load_yaml_config(self) -> Dict:
        """
        加载YAML配置文件
        
        Returns:
            配置字典
        """
        print(f"\n加载YAML配置...")
        
        if not os.path.exists(self.yaml_file):
            raise FileNotFoundError(f"YAML配置文件不存在: {self.yaml_file}")
        
        with open(self.yaml_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 提取ROI坐标
        roi_coords = config.get('roi_coordinates', {})
        
        print(f"  ROI坐标:")
        print(f"    X: [{roi_coords['x']['min']}, {roi_coords['x']['max']}] (size: {roi_coords['x']['size']})")
        print(f"    Y: [{roi_coords['y']['min']}, {roi_coords['y']['max']}] (size: {roi_coords['y']['size']})")
        print(f"    Z: [{roi_coords['z']['min']}, {roi_coords['z']['max']}] (size: {roi_coords['z']['size']})")
        
        return config
    
    def load_dicom_series(self) -> Tuple[np.ndarray, Dict]:
        """
        加载DICOM序列
        
        Returns:
            (体块数组, DICOM元数据)
        """
        print(f"\n加载DICOM序列...")
        
        # 获取所有DICOM文件
        dcm_files = sorted([
            os.path.join(self.dcm_folder, f) 
            for f in os.listdir(self.dcm_folder) 
            if f.endswith('.dcm')
        ])
        
        if len(dcm_files) == 0:
            raise ValueError(f"在 {self.dcm_folder} 中未找到DICOM文件")
        
        print(f"  找到 {len(dcm_files)} 个DICOM文件")
        
        # 读取第一个文件获取基本信息
        first_dcm = pydicom.dcmread(dcm_files[0])
        
        # 提取图像尺寸
        rows = int(first_dcm.Rows)
        cols = int(first_dcm.Columns)
        num_slices = len(dcm_files)
        
        print(f"  图像尺寸: {rows} × {cols} × {num_slices}")
        
        # 初始化体块数组
        volume = np.zeros((num_slices, rows, cols), dtype=np.int16)
        
        # 读取所有切片
        print(f"  读取切片...")
        for i, dcm_file in enumerate(tqdm(dcm_files, desc="  加载DICOM")):
            dcm = pydicom.dcmread(dcm_file)
            pixel_array = dcm.pixel_array
            
            # 应用Y轴翻转（与dcmprocess.py保持一致）
            pixel_array = np.flipud(pixel_array)
            
            volume[i] = pixel_array
        
        # 提取元数据
        metadata = {
            'PixelSpacing': first_dcm.PixelSpacing if hasattr(first_dcm, 'PixelSpacing') else [1.0, 1.0],
            'SliceThickness': first_dcm.SliceThickness if hasattr(first_dcm, 'SliceThickness') else 1.0,
            'RescaleIntercept': first_dcm.RescaleIntercept if hasattr(first_dcm, 'RescaleIntercept') else 0,
            'RescaleSlope': first_dcm.RescaleSlope if hasattr(first_dcm, 'RescaleSlope') else 1,
        }
        
        print(f"  元数据:")
        print(f"    PixelSpacing: {metadata['PixelSpacing']}")
        print(f"    SliceThickness: {metadata['SliceThickness']}")
        print(f"    RescaleIntercept: {metadata['RescaleIntercept']}")
        print(f"    RescaleSlope: {metadata['RescaleSlope']}")
        
        return volume, metadata
    
    def crop_volume(self, volume: np.ndarray, roi_coords: Dict) -> np.ndarray:
        """
        根据ROI坐标裁剪体块
        
        Args:
            volume: 原始体块数组 (Z, Y, X)
            roi_coords: ROI坐标字典
        
        Returns:
            裁剪后的体块
        """
        print(f"\n裁剪体块...")
        
        # 提取坐标
        x_min = roi_coords['x']['min']
        x_max = roi_coords['x']['max']
        y_min = roi_coords['y']['min']
        y_max = roi_coords['y']['max']
        z_min = roi_coords['z']['min']
        z_max = roi_coords['z']['max']
        
        print(f"  原始体块形状: {volume.shape}")
        print(f"  裁剪范围:")
        print(f"    X: [{x_min}, {x_max}]")
        print(f"    Y: [{y_min}, {y_max}]")
        print(f"    Z: [{z_min}, {z_max}]")
        
        # 裁剪 (注意DICOM通常是ZYX顺序)
        cropped = volume[z_min:z_max, y_min:y_max, x_min:x_max]
        
        print(f"  裁剪后形状: {cropped.shape}")
        print(f"  数值范围: [{cropped.min()}, {cropped.max()}]")
        
        return cropped
    
    def save_as_nifti(self, volume: np.ndarray, metadata: Dict, output_name: str = "cropped_volume.nii.gz"):
        """
        保存为NIFTI格式
        
        Args:
            volume: 体块数组 (Z, Y, X)
            metadata: DICOM元数据
            output_name: 输出文件名
        """
        print(f"\n保存为NIFTI格式...")
        print(f"  输入形状: {volume.shape} (Z, Y, X)")
        
        # 关键修复: 维度转置 (Z, Y, X) -> (X, Y, Z) 以符合NIFTI标准
        volume_transposed = np.transpose(volume, (2, 1, 0))
        print(f"  转置后形状: {volume_transposed.shape} (X, Y, Z)")
        
        # 创建仿射矩阵（与dcmcompose.py保持一致）
        pixel_spacing = metadata['PixelSpacing']
        slice_thickness = metadata['SliceThickness']
        
        affine = np.eye(4)
        affine[0, 0] = pixel_spacing[1]  # 列间距 (x)
        affine[1, 1] = pixel_spacing[0]  # 行间距 (y)
        affine[2, 2] = slice_thickness    # 切片间距 (z)
        
        # 使用转置后的数据创建NIFTI图像
        nifti_img = nib.Nifti1Image(volume_transposed, affine)
        
        # 保存
        output_path = os.path.join(self.output_folder, output_name)
        nib.save(nifti_img, output_path)
        
        print(f"  ✓ 已保存到: {output_path}")
        print(f"  文件大小: {os.path.getsize(output_path) / (1024**2):.2f} MB")
    
    def process(self, output_name: str = "cropped_volume.nii.gz"):
        """
        执行完整的裁剪流程
        
        Args:
            output_name: 输出文件名
        """
        print("="*60)
        print("DICOM体块裁剪工具")
        print("="*60)
        
        try:
            # 1. 加载YAML配置
            config = self.load_yaml_config()
            roi_coords = config['roi_coordinates']
            
            # 2. 加载DICOM序列
            volume, metadata = self.load_dicom_series()
            
            # 3. 裁剪体块
            cropped_volume = self.crop_volume(volume, roi_coords)
            
            # 4. 保存为NIFTI
            self.save_as_nifti(cropped_volume, metadata, output_name)
            
            print("\n" + "="*60)
            print("裁剪完成!")
            print("="*60)
            
        except Exception as e:
            print(f"\n错误: {e}")
            import traceback
            traceback.print_exc()
            raise


def parse_args():
    parser = argparse.ArgumentParser(
        description="DICOM体块裁剪工具 - 根据YAML配置裁剪ROI并保存为NIFTI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "-d", "--dicom-folder",
        required=True,
        help="DICOM文件夹路径"
    )
    parser.add_argument(
        "-y", "--yaml-file",
        required=True,
        help="YAML配置文件路径（包含ROI坐标）"
    )
    parser.add_argument(
        "-o", "--output-folder",
        default="./cropped_nifti",
        help="输出NIFTI文件夹"
    )
    parser.add_argument(
        "-n", "--output-name",
        default="cropped_volume.nii.gz",
        help="输出NIFTI文件名"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 检查输入
    if not os.path.exists(args.dicom_folder):
        print(f"错误: DICOM文件夹不存在: {args.dicom_folder}")
        sys.exit(1)
    
    if not os.path.exists(args.yaml_file):
        print(f"错误: YAML文件不存在: {args.yaml_file}")
        sys.exit(1)
    
    # 创建裁剪器并执行
    cropper = DicomCropper(
        dcm_folder=args.dicom_folder,
        yaml_file=args.yaml_file,
        output_folder=args.output_folder
    )
    
    cropper.process(output_name=args.output_name)


if __name__ == "__main__":
    main()