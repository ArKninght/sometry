#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量组合工具
对文件夹下的多个ROI NIFTI文件进行批量组合，将它们插入到原始DICOM体块中并保存为RAW文件
"""

import argparse
import os
from pathlib import Path
from typing import List

import nibabel as nib
import numpy as np
import yaml

# 导入原始的VolumeComposer类
from compose4Dandraw import VolumeComposer


class BatchComposer:
    """批量组合处理器"""
    
    def __init__(self, roi_dir: str, dicom_folder: str, yaml_path: str, 
                 output_folder: str, dtype: str = 'int16'):
        """
        初始化批量组合器
        
        Args:
            roi_dir: ROI NIFTI文件目录
            dicom_folder: 原始DICOM文件夹路径
            yaml_path: ROI坐标配置YAML文件路径
            output_folder: 输出RAW文件的文件夹路径
            dtype: 输出数据类型
        """
        self.roi_dir = Path(roi_dir)
        self.dicom_folder = dicom_folder
        self.yaml_path = yaml_path
        self.output_folder = Path(output_folder)
        self.dtype = dtype
        
        # 创建输出目录
        self.output_folder.mkdir(exist_ok=True, parents=True)
    
    def find_roi_files(self, pattern: str = "*.nii.gz") -> List[Path]:
        """
        查找所有匹配的ROI NIFTI文件
        
        Args:
            pattern: 文件匹配模式
        
        Returns:
            ROI文件路径列表
        """
        print(f"在目录 {self.roi_dir} 中查找ROI NIFTI文件...")
        print(f"匹配模式: {pattern}")
        
        roi_files = list(self.roi_dir.glob(pattern))
        roi_files.sort()  # 排序以确保处理顺序
        
        print(f"找到 {len(roi_files)} 个ROI文件:")
        for i, f in enumerate(roi_files, 1):
            print(f"  {i}. {f.name}")
        
        return roi_files
    
    def process_single_file(self, roi_path: Path) -> Path:
        """
        处理单个ROI文件
        
        Args:
            roi_path: ROI文件路径
        
        Returns:
            输出RAW文件路径
        """
        # 生成输出文件名（移除扩展名）
        output_filename = roi_path.stem  # 移除.nii.gz
        if output_filename.endswith('.nii'):
            output_filename = output_filename[:-4]  # 移除.nii
        
        print(f"\n{'='*60}")
        print(f"处理文件: {roi_path.name}")
        print(f"输出文件名: {output_filename}")
        print(f"{'='*60}")
        
        # 创建组合器
        composer = VolumeComposer(
            roi_nifti_path=str(roi_path),
            dicom_folder=self.dicom_folder,
            yaml_path=self.yaml_path,
            output_folder=str(self.output_folder)
        )
        
        # 执行组合
        output_path = composer.compose(
            output_filename=output_filename,
            dtype=self.dtype
        )
        
        return Path(output_path)
    
    def process_all(self, pattern: str = "*.nii.gz") -> List[Path]:
        """
        批量处理所有文件
        
        Args:
            pattern: 文件匹配模式
        
        Returns:
            输出文件路径列表
        """
        print("="*70)
        print("批量ROI组合处理")
        print("="*70)
        
        # 查找所有ROI文件
        roi_files = self.find_roi_files(pattern)
        
        if not roi_files:
            print("\n警告: 未找到匹配的ROI文件!")
            return []
        
        # 显示处理配置
        print(f"\n处理配置:")
        print(f"  DICOM目录: {self.dicom_folder}")
        print(f"  YAML配置: {self.yaml_path}")
        print(f"  输出目录: {self.output_folder}")
        print(f"  数据类型: {self.dtype}")
        
        # 批量处理
        output_files = []
        success_count = 0
        failed_files = []
        
        for i, roi_file in enumerate(roi_files, 1):
            try:
                print(f"\n进度: {i}/{len(roi_files)}")
                output_file = self.process_single_file(roi_file)
                output_files.append(output_file)
                success_count += 1
                print(f"✓ 成功处理: {roi_file.name}")
                
            except Exception as e:
                print(f"✗ 处理失败: {roi_file.name}")
                print(f"  错误: {e}")
                failed_files.append((roi_file, str(e)))
        
        # 输出总结
        print("\n" + "="*70)
        print("批量处理完成")
        print("="*70)
        print(f"总文件数: {len(roi_files)}")
        print(f"成功: {success_count}")
        print(f"失败: {len(failed_files)}")
        
        if failed_files:
            print("\n失败的文件:")
            for file, error in failed_files:
                print(f"  - {file.name}: {error}")
        
        print(f"\n输出目录: {self.output_folder}")
        print("="*70)
        
        return output_files
    
    def generate_summary_report(self, output_files: List[Path]):
        """
        生成处理汇总报告
        
        Args:
            output_files: 输出文件列表
        """
        report_path = self.output_folder / "compose_summary.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("批量ROI组合处理汇总报告\n")
            f.write("="*70 + "\n\n")
            
            f.write("处理配置:\n")
            f.write(f"  ROI目录: {self.roi_dir}\n")
            f.write(f"  DICOM目录: {self.dicom_folder}\n")
            f.write(f"  YAML配置: {self.yaml_path}\n")
            f.write(f"  输出目录: {self.output_folder}\n")
            f.write(f"  数据类型: {self.dtype}\n")
            f.write("\n")
            
            f.write(f"处理结果:\n")
            f.write(f"  总文件数: {len(output_files)}\n")
            f.write("\n")
            
            f.write("输出文件列表:\n")
            for i, output_file in enumerate(output_files, 1):
                # 获取文件大小
                if output_file.exists():
                    size_mb = output_file.stat().st_size / (1024 * 1024)
                    f.write(f"  {i}. {output_file.name} ({size_mb:.2f} MB)\n")
                else:
                    f.write(f"  {i}. {output_file.name} (文件不存在)\n")
            
            f.write("\n" + "="*70 + "\n")
        
        print(f"\n汇总报告已保存: {report_path}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="批量将多个ROI体块插入到原始DICOM体块中并保存为RAW文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 批量处理目录下所有ROI文件
  python batch_compose.py \\
      -i VoxelComposeMice/4DCT_output_rawrange \\
      -d /mnt/d/多帧扫描/recon/recon2/fusion/ \\
      -y VoxelComposeMice/nifti_output/hu_adjustment_stats.yaml \\
      -o VoxelComposeMice/composed_output
  
  # 只处理特定模式的文件
  python batch_compose.py \\
      -i VoxelComposeMice/4DCT_output_rawrange \\
      -d /mnt/d/多帧扫描/recon/recon2/fusion/ \\
      -y VoxelComposeMice/nifti_output/hu_adjustment_stats.yaml \\
      -o VoxelComposeMice/composed_output \\
      --pattern "phase_*.nii.gz"
  
  # 指定输出数据类型
  python batch_compose.py \\
      -i VoxelComposeMice/4DCT_output_rawrange \\
      -d /mnt/d/多帧扫描/recon/recon2/fusion/ \\
      -y VoxelComposeMice/nifti_output/hu_adjustment_stats.yaml \\
      -o VoxelComposeMice/composed_output \\
      --dtype float32
        """
    )
    
    parser.add_argument(
        "-i", "--roi-dir",
        required=True,
        help="ROI NIFTI文件目录"
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
        "--dtype",
        choices=['float32', 'float64', 'int16', 'uint16', 'uint8', 'int32', 'uint32'],
        default='int16',
        help="RAW文件的数据类型（默认: int16）"
    )
    parser.add_argument(
        "--pattern",
        default="*.nii.gz",
        help="文件匹配模式（默认: *.nii.gz）"
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="不生成汇总报告"
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 检查输入目录和文件
    if not os.path.exists(args.roi_dir):
        print(f"错误: ROI目录不存在: {args.roi_dir}")
        return
    
    if not os.path.exists(args.dicom_folder):
        print(f"错误: DICOM文件夹不存在: {args.dicom_folder}")
        return
    
    if not os.path.exists(args.yaml):
        print(f"错误: YAML配置文件不存在: {args.yaml}")
        return
    
    try:
        # 创建批量组合器
        composer = BatchComposer(
            roi_dir=args.roi_dir,
            dicom_folder=args.dicom_folder,
            yaml_path=args.yaml,
            output_folder=args.output_folder,
            dtype=args.dtype
        )
        
        # 批量处理
        output_files = composer.process_all(pattern=args.pattern)
        
        # 生成汇总报告
        if not args.no_report and output_files:
            composer.generate_summary_report(output_files)
        
    except Exception as e:
        print(f"\n批量处理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()