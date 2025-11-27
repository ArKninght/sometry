#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量上采样工具
对文件夹下的所有NIFTI文件进行批量上采样处理
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from PIL import Image

# 导入原始的VolumeUpsampler类
from upsampling import VolumeUpsampler


class BatchUpsampler:
    """批量上采样处理器"""
    
    def __init__(self, input_dir: str, output_dir: str, 
                 target_shape: Tuple[int, int, int] = (512, 512, 256),
                 method: str = "lanczos3"):
        """
        初始化批量上采样器
        
        Args:
            input_dir: 输入NIFTI文件目录
            output_dir: 输出目录
            target_shape: 目标形状 (x, y, z)
            method: 插值方法
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.target_shape = target_shape
        self.method = method
        
        # 创建输出目录
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # 创建上采样器
        self.upsampler = VolumeUpsampler(target_shape)
    
    def find_nifti_files(self, pattern: str = "*.nii.gz") -> List[Path]:
        """
        查找所有匹配的NIFTI文件
        
        Args:
            pattern: 文件匹配模式
        
        Returns:
            NIFTI文件路径列表
        """
        print(f"在目录 {self.input_dir} 中查找NIFTI文件...")
        print(f"匹配模式: {pattern}")
        
        nifti_files = list(self.input_dir.glob(pattern))
        nifti_files.sort()  # 排序以确保处理顺序
        
        print(f"找到 {len(nifti_files)} 个NIFTI文件:")
        for i, f in enumerate(nifti_files, 1):
            print(f"  {i}. {f.name}")
        
        return nifti_files
    
    def process_single_file(self, input_path: Path) -> Path:
        """
        处理单个NIFTI文件
        
        Args:
            input_path: 输入文件路径
        
        Returns:
            输出文件路径
        """
        # 生成输出文件名
        output_filename = input_path.stem  # 移除.nii.gz
        if output_filename.endswith('.nii'):
            output_filename = output_filename[:-4]  # 移除.nii
        
        # 添加后缀标识已上采样
        output_filename = f"{output_filename}_{self.target_shape[0]}.nii.gz"
        output_path = self.output_dir / output_filename
        
        print(f"\n{'='*60}")
        print(f"处理文件: {input_path.name}")
        print(f"输出到: {output_path.name}")
        print(f"{'='*60}")
        
        # 使用上采样器处理
        self.upsampler.process(
            str(input_path),
            str(output_path),
            self.method
        )
        
        return output_path
    
    def process_all(self, pattern: str = "*.nii.gz") -> List[Path]:
        """
        批量处理所有文件
        
        Args:
            pattern: 文件匹配模式
        
        Returns:
            输出文件路径列表
        """
        print("="*70)
        print("批量NIFTI上采样处理")
        print("="*70)
        
        # 查找所有文件
        input_files = self.find_nifti_files(pattern)
        
        if not input_files:
            print("\n警告: 未找到匹配的NIFTI文件!")
            return []
        
        # 显示处理配置
        print(f"\n处理配置:")
        print(f"  目标形状: {self.target_shape}")
        print(f"  插值方法: {self.method}")
        print(f"  输出目录: {self.output_dir}")
        
        # 批量处理
        output_files = []
        success_count = 0
        failed_files = []
        
        for i, input_file in enumerate(input_files, 1):
            try:
                print(f"\n进度: {i}/{len(input_files)}")
                output_file = self.process_single_file(input_file)
                output_files.append(output_file)
                success_count += 1
                print(f"✓ 成功处理: {input_file.name}")
                
            except Exception as e:
                print(f"✗ 处理失败: {input_file.name}")
                print(f"  错误: {e}")
                failed_files.append((input_file, str(e)))
        
        # 输出总结
        print("\n" + "="*70)
        print("批量处理完成")
        print("="*70)
        print(f"总文件数: {len(input_files)}")
        print(f"成功: {success_count}")
        print(f"失败: {len(failed_files)}")
        
        if failed_files:
            print("\n失败的文件:")
            for file, error in failed_files:
                print(f"  - {file.name}: {error}")
        
        print(f"\n输出目录: {self.output_dir}")
        print("="*70)
        
        return output_files
    
    def generate_summary_report(self, output_files: List[Path]):
        """
        生成处理汇总报告
        
        Args:
            output_files: 输出文件列表
        """
        report_path = self.output_dir / "upsampling_summary.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("批量NIFTI上采样处理汇总报告\n")
            f.write("="*70 + "\n\n")
            
            f.write("处理配置:\n")
            f.write(f"  输入目录: {self.input_dir}\n")
            f.write(f"  输出目录: {self.output_dir}\n")
            f.write(f"  目标形状: {self.target_shape}\n")
            f.write(f"  插值方法: {self.method}\n")
            f.write("\n")
            
            f.write(f"处理结果:\n")
            f.write(f"  总文件数: {len(output_files)}\n")
            f.write("\n")
            
            f.write("输出文件列表:\n")
            for i, output_file in enumerate(output_files, 1):
                # 获取文件大小
                size_mb = output_file.stat().st_size / (1024 * 1024)
                f.write(f"  {i}. {output_file.name} ({size_mb:.2f} MB)\n")
            
            f.write("\n" + "="*70 + "\n")
        
        print(f"\n汇总报告已保存: {report_path}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="批量上采样多个NIFTI文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 批量上采样到512x512x256（默认）
  python batch_upsampling.py \\
      -i VoxelComposeMice/4DCT_NIFIT_split \\
      -o VoxelComposeMice/4DCT_NIFIT_upsample
  
  # 使用不同的目标尺寸
  python batch_upsampling.py \\
      -i VoxelComposeMice/4DCT_NIFIT_split \\
      -o VoxelComposeMice/4DCT_NIFIT_upsample \\
      -s 1024 1024 512
  
  # 使用不同的插值方法
  python batch_upsampling.py \\
      -i VoxelComposeMice/4DCT_NIFIT_split \\
      -o VoxelComposeMice/4DCT_NIFIT_upsample \\
      -m cubic
  
  # 只处理特定模式的文件
  python batch_upsampling.py \\
      -i VoxelComposeMice/4DCT_NIFIT_split \\
      -o VoxelComposeMice/4DCT_NIFIT_upsample \\
      --pattern "phase_*.nii.gz"
        """
    )
    
    parser.add_argument(
        "-i", "--input-dir",
        required=True,
        help="输入NIFTI文件目录"
    )
    parser.add_argument(
        "-o", "--output-dir",
        required=True,
        help="输出目录"
    )
    parser.add_argument(
        "-s", "--size",
        nargs=3,
        type=int,
        default=[512, 512, 256],
        help="目标尺寸 (x y z)，默认: 512 512 256"
    )
    parser.add_argument(
        "-m", "--method",
        choices=["nearest", "linear", "cubic", "lanczos3"],
        default="nearest",
        help="插值方法（默认: nearest）"
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
    
    # 检查输入目录
    if not os.path.exists(args.input_dir):
        print(f"错误: 输入目录不存在: {args.input_dir}")
        return
    
    try:
        # 创建批量上采样器
        upsampler = BatchUpsampler(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            target_shape=tuple(args.size),
            method=args.method
        )
        
        # 批量处理
        output_files = upsampler.process_all(pattern=args.pattern)
        
        # 生成汇总报告
        if not args.no_report and output_files:
            upsampler.generate_summary_report(output_files)
        
    except Exception as e:
        print(f"\n批量处理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()