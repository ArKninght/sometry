#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NIfTI文件格式转换工具
支持将NIfTI图像转换为:
1. DICOM序列 (每个切片一个DICOM文件)
2. RAW文件 (单一二进制文件)
"""

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import nibabel as nib
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaInformation
from pydicom.uid import generate_uid
from datetime import datetime


class NIfTIConverter:
    """NIfTI文件格式转换器"""
    
    def __init__(self, input_path: str, output_dir: str):
        self.input_path = input_path
        self.output_dir = output_dir
        self.volume = None
        self.affine = None
        self.header = None
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
    def load_nifti(self) -> np.ndarray:
        """加载NIfTI文件"""
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"输入文件不存在: {self.input_path}")
        
        print(f"加载NIfTI文件: {self.input_path}")
        img = nib.load(self.input_path)
        self.volume = img.get_fdata(dtype=np.float32)
        self.affine = img.affine
        self.header = img.header
        
        print(f"  形状: {self.volume.shape}")
        print(f"  数据类型: {self.volume.dtype}")
        print(f"  数值范围: [{self.volume.min():.2f}, {self.volume.max():.2f}]")
        print(f"  体素尺寸: {self.header.get_zooms()[:3]} mm")
        
        return self.volume
    
    def to_raw(self, dtype: str = 'float32', order: str = 'C') -> str:
        """
        转换为RAW文件
        
        Args:
            dtype: 输出数据类型 ('float32', 'float64', 'int16', 'uint16', 'uint8')
            order: 存储顺序 ('C'行优先, 'F'列优先)
        
        Returns:
            输出文件路径
        """
        if self.volume is None:
            self.load_nifti()
        
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
            raise ValueError(f"不支持的数据类型: {dtype}. 支持的类型: {list(dtype_map.keys())}")
        
        # 转换数据类型
        output_volume = self.volume.astype(dtype_map[dtype])
        
        # 生成输出文件名
        input_name = Path(self.input_path).stem
        if input_name.endswith('.nii'):
            input_name = input_name[:-4]
        output_path = os.path.join(self.output_dir, f"{input_name}_{dtype}.raw")
        
        print(f"\n转换为RAW文件...")
        print(f"  输出类型: {dtype}")
        print(f"  存储顺序: {order}")
        print(f"  输出路径: {output_path}")
        
        # 保存为RAW文件
        output_volume.tofile(output_path, sep='', format=order)
        
        # 保存元数据信息文件
        meta_path = output_path.replace('.raw', '_info.txt')
        with open(meta_path, 'w') as f:
            f.write(f"RAW文件信息\n")
            f.write(f"{'='*50}\n")
            f.write(f"源文件: {self.input_path}\n")
            f.write(f"形状: {output_volume.shape} (Z, Y, X)\n")
            f.write(f"数据类型: {dtype}\n")
            f.write(f"存储顺序: {'行优先(C)' if order == 'C' else '列优先(F)'}\n")
            f.write(f"体素尺寸: {self.header.get_zooms()[:3]} mm\n")
            f.write(f"数值范围: [{output_volume.min():.2f}, {output_volume.max():.2f}]\n")
            f.write(f"文件大小: {os.path.getsize(output_path) / (1024**2):.2f} MB\n")
        
        print(f"  RAW文件已保存")
        print(f"  元数据已保存: {meta_path}")
        
        return output_path
    
    def to_dicom(
        self, 
        patient_name: str = "Anonymous",
        patient_id: str = "000000",
        study_description: str = "CT Study",
        series_description: str = "CT Series"
    ) -> str:
        """
        转换为DICOM序列
        
        Args:
            patient_name: 患者姓名
            patient_id: 患者ID
            study_description: 检查描述
            series_description: 序列描述
        
        Returns:
            输出目录路径
        """
        if self.volume is None:
            self.load_nifti()
        
        # 创建DICOM输出子目录
        input_name = Path(self.input_path).stem
        if input_name.endswith('.nii'):
            input_name = input_name[:-4]
        dicom_dir = os.path.join(self.output_dir, f"{input_name}_dicom")
        os.makedirs(dicom_dir, exist_ok=True)
        
        print(f"\n转换为DICOM序列...")
        print(f"  输出目录: {dicom_dir}")
        
        # 获取体素尺寸
        pixel_spacing = self.header.get_zooms()
        
        # 生成唯一标识符
        study_instance_uid = generate_uid()
        series_instance_uid = generate_uid()
        frame_of_reference_uid = generate_uid()
        
        # 当前时间
        dt = datetime.now()
        date_str = dt.strftime('%Y%m%d')
        time_str = dt.strftime('%H%M%S')
        
        # 数据归一化到int16范围
        volume_normalized = self._normalize_to_int16(self.volume)
        
        # 逐切片保存为DICOM
        num_slices = volume_normalized.shape[0]
        print(f"  切片数量: {num_slices}")
        
        for slice_idx in range(num_slices):
            # 创建DICOM数据集
            ds = self._create_dicom_dataset(
                slice_data=volume_normalized[slice_idx, :, :],
                slice_idx=slice_idx,
                num_slices=num_slices,
                patient_name=patient_name,
                patient_id=patient_id,
                study_description=study_description,
                series_description=series_description,
                study_instance_uid=study_instance_uid,
                series_instance_uid=series_instance_uid,
                frame_of_reference_uid=frame_of_reference_uid,
                pixel_spacing=pixel_spacing,
                date_str=date_str,
                time_str=time_str
            )
            
            # 保存DICOM文件
            output_filename = os.path.join(dicom_dir, f"slice_{slice_idx:04d}.dcm")
            ds.save_as(output_filename, write_like_original=False)
            
            if (slice_idx + 1) % 50 == 0 or slice_idx == num_slices - 1:
                print(f"  已保存: {slice_idx + 1}/{num_slices} 切片")
        
        print(f"  DICOM序列已保存到: {dicom_dir}")
        return dicom_dir
    
    def _normalize_to_int16(self, volume: np.ndarray) -> np.ndarray:
        """将数据归一化到int16范围"""
        vmin, vmax = volume.min(), volume.max()
        
        if vmax - vmin < 1e-6:
            return np.zeros(volume.shape, dtype=np.int16)
        
        # 映射到[-32768, 32767]范围
        normalized = ((volume - vmin) / (vmax - vmin) * 65535 - 32768).astype(np.int16)
        
        print(f"  数据归一化: [{vmin:.2f}, {vmax:.2f}] → [-32768, 32767]")
        return normalized
    
    def _create_dicom_dataset(
        self,
        slice_data: np.ndarray,
        slice_idx: int,
        num_slices: int,
        patient_name: str,
        patient_id: str,
        study_description: str,
        series_description: str,
        study_instance_uid: str,
        series_instance_uid: str,
        frame_of_reference_uid: str,
        pixel_spacing: Tuple[float, float, float],
        date_str: str,
        time_str: str
    ) -> Dataset:
        """创建DICOM数据集"""
        
        # 创建文件元信息
        file_meta = FileMetaInformation()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1'  # Explicit VR Little Endian
        file_meta.ImplementationClassUID = generate_uid()
        
        # 创建数据集
        ds = Dataset()
        ds.file_meta = file_meta
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        
        # 患者信息
        ds.PatientName = patient_name
        ds.PatientID = patient_id
        ds.PatientBirthDate = ''
        ds.PatientSex = ''
        
        # 检查信息
        ds.StudyInstanceUID = study_instance_uid
        ds.StudyDate = date_str
        ds.StudyTime = time_str
        ds.StudyDescription = study_description
        ds.StudyID = '1'
        
        # 序列信息
        ds.SeriesInstanceUID = series_instance_uid
        ds.SeriesNumber = 1
        ds.SeriesDescription = series_description
        ds.Modality = 'CT'
        
        # 图像信息
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.InstanceNumber = slice_idx + 1
        
        # 图像方向和位置
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [0, 0, slice_idx * pixel_spacing[2]]
        ds.FrameOfReferenceUID = frame_of_reference_uid
        
        # 像素数据属性
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = 'MONOCHROME2'
        ds.Rows = slice_data.shape[0]
        ds.Columns = slice_data.shape[1]
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1  # signed
        
        # 体素尺寸
        ds.PixelSpacing = [float(pixel_spacing[1]), float(pixel_spacing[0])]
        ds.SliceThickness = float(pixel_spacing[2])
        
        # 像素数据
        ds.PixelData = slice_data.tobytes()
        
        return ds


def parse_args():
    parser = argparse.ArgumentParser(
        description="NIfTI文件格式转换工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="输入NIfTI文件路径 (.nii 或 .nii.gz)"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="输出目录路径"
    )
    parser.add_argument(
        "-f", "--format",
        choices=['dicom', 'raw', 'both'],
        default='dicom',
        help="输出格式: dicom(DICOM序列), raw(RAW文件), both(两者都输出)"
    )
    parser.add_argument(
        "--dtype",
        choices=['float32', 'float64', 'int16', 'uint16', 'uint8', 'int32', 'uint32'],
        default='float32',
        help="RAW文件的数据类型"
    )
    parser.add_argument(
        "--order",
        choices=['C', 'F'],
        default='C',
        help="RAW文件的存储顺序: C(行优先), F(列优先)"
    )
    parser.add_argument(
        "--patient-name",
        default="Anonymous",
        help="DICOM患者姓名"
    )
    parser.add_argument(
        "--patient-id",
        default="000000",
        help="DICOM患者ID"
    )
    parser.add_argument(
        "--study-desc",
        default="CT Study",
        help="DICOM检查描述"
    )
    parser.add_argument(
        "--series-desc",
        default="CT Series",
        help="DICOM序列描述"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*60)
    print("NIfTI文件格式转换工具")
    print("="*60)
    
    converter = NIfTIConverter(args.input, args.output)
    converter.load_nifti()
    
    if args.format in ['raw', 'both']:
        converter.to_raw(dtype=args.dtype, order=args.order)
    
    if args.format in ['dicom', 'both']:
        converter.to_dicom(
            patient_name=args.patient_name,
            patient_id=args.patient_id,
            study_description=args.study_desc,
            series_description=args.series_desc
        )
    
    print("\n" + "="*60)
    print("转换完成!")
    print("="*60)


if __name__ == "__main__":
    main()