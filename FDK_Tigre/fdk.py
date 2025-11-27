#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TIGRE FDK投影模拟工具
读取RAW格式的体块数据,使用TIGRE生成360个角度的模拟投影数据
"""

import argparse
import os
from typing import Tuple, Optional

import numpy as np
import tigre
import tigre.algorithms as algs
from matplotlib import pyplot as plt


class TigreProjectionSimulator:
    """TIGRE投影模拟器"""
    
    def __init__(self, raw_file: str, volume_shape: Tuple[int, int, int], 
                 dtype: str = 'float32', output_dir: str = './projection_output'):
        """
        初始化投影模拟器
        
        Args:
            raw_file: RAW文件路径
            volume_shape: 体块形状 (Z, Y, X) 对应TIGRE的(nVoxel[0], nVoxel[1], nVoxel[2])
            dtype: 数据类型
            output_dir: 输出目录
        """
        self.raw_file = raw_file
        self.volume_shape = volume_shape
        self.dtype = dtype
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化几何配置(基于tryTigre01.py)
        self.geo = self._init_geometry()
        
    def _init_geometry(self):
        """
        初始化TIGRE几何配置
        参考tryTigre01.py中的设置,但根据实际体块大小调整
        """
        geo = tigre.geometry()
        
        # 距离参数
        geo.DSD = 1566.30  # 源到探测器距离 (mm)
        geo.DSO = 1566.30  # 源到旋转中心距离 (mm)
        
        # 探测器参数
        geo.nDetector = np.array([768, 1032])  # 探测器像素数 (px)
        geo.dDetector = np.array([1.0, 1.0])  # 像素大小 (mm)
        geo.sDetector = geo.nDetector * geo.dDetector  # 探测器总尺寸 (mm)
        
        # 体块参数 - 根据输入数据动态设置
        geo.nVoxel = np.array([self.volume_shape[0], 
                              self.volume_shape[1], 
                              self.volume_shape[2]])  # 体素数量 (vx)
        
        # 体素大小 - 根据体块大小调整
        # 假设物理尺寸与原配置相似,计算相应的体素大小
        reference_voxel = np.array([1.0, 1.0, 1.0])  # 参考体素大小
        
        # 保持物理尺寸不变,计算新的体素大小
        geo.dVoxel = reference_voxel
        geo.sVoxel = geo.dVoxel * geo.nVoxel  # 体块总尺寸 (mm)
        
        # 偏移量
        geo.offOrigin = np.array([0, 0, 0])  # 图像原点偏移 (mm)
        geo.offDetector = np.array([0, 0])  # 探测器偏移 (mm)
        
        # 辅助参数
        geo.accuracy = 0.2  # 插值投影精度
        
        # 可选参数
        geo.COR = 0  # Y方向位移
        geo.rotDetector = np.array([0, 0, 0])  # 探测器旋转
        geo.mode = "cone"  # 锥束几何
        
        print("TIGRE几何配置:")
        print(f"  体块形状: {geo.nVoxel} (Z×Y×X)")
        print(f"  体素大小: {geo.dVoxel} mm")
        print(f"  体块物理尺寸: {geo.sVoxel} mm")
        print(f"  探测器像素: {geo.nDetector}")
        print(f"  探测器像素大小: {geo.dDetector} mm")
        print(f"  DSD: {geo.DSD} mm, DSO: {geo.DSO} mm")
        
        return geo
    
    def load_raw_volume(self) -> np.ndarray:
        """加载RAW格式的体块数据"""
        print(f"\n加载RAW文件: {self.raw_file}")
        
        if not os.path.exists(self.raw_file):
            raise FileNotFoundError(f"文件不存在: {self.raw_file}")
        
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
        
        if self.dtype not in dtype_map:
            raise ValueError(f"不支持的数据类型: {self.dtype}")
        
        # 读取RAW文件
        volume = np.fromfile(self.raw_file, dtype=dtype_map[self.dtype])
        
        # 重塑为3D数组
        expected_size = np.prod(self.volume_shape)
        if volume.size != expected_size:
            raise ValueError(
                f"文件大小不匹配: 期望 {expected_size} 个元素, "
                f"实际 {volume.size} 个元素"
            )
        
        volume = volume.reshape(self.volume_shape)
        
        print(f"  形状: {volume.shape} (Z, Y, X)")
        print(f"  数据类型: {volume.dtype}")
        print(f"  数值范围: [{volume.min():.2f}, {volume.max():.2f}]")
        print(f"  文件大小: {os.path.getsize(self.raw_file) / (1024**2):.2f} MB")
        
        # 转换为float32供TIGRE使用
        if volume.dtype != np.float32:
            print(f"  转换为float32...")
            volume = volume.astype(np.float32)
        
        return volume
    
    def generate_projections(self, volume: np.ndarray, 
                           num_angles: int = 360,
                           angle_range: Tuple[float, float] = (0, 2*np.pi)) -> np.ndarray:
        """
        生成模拟投影数据
        
        Args:
            volume: 3D体块数据
            num_angles: 投影角度数量
            angle_range: 角度范围(弧度) (start, end)
        
        Returns:
            投影数据数组 (num_angles, detector_rows, detector_cols)
        """
        print(f"\n生成投影数据...")
        print(f"  角度数量: {num_angles}")
        print(f"  角度范围: {angle_range[0]:.2f} - {angle_range[1]:.2f} 弧度")
        print(f"  角度范围: {np.degrees(angle_range[0]):.1f}° - {np.degrees(angle_range[1]):.1f}°")
        
        # 定义投影角度
        angles = np.linspace(angle_range[0], angle_range[1], num_angles)
        
        # 使用TIGRE的Ax算子进行投影
        print(f"  正在计算投影 (使用TIGRE Ax算子)...")
        projections = tigre.Ax(volume, self.geo, angles)
        
        print(f"  投影数据形状: {projections.shape}")
        print(f"  投影数值范围: [{projections.min():.4f}, {projections.max():.4f}]")
        
        return projections, angles
    
    def save_projections(self, projections: np.ndarray, angles: np.ndarray,
                        save_individual: bool = True,
                        save_stack: bool = True):
        """
        保存投影数据
        
        Args:
            projections: 投影数据
            angles: 角度数组
            save_individual: 是否保存单独的投影帧
            save_stack: 是否保存完整的投影栈
        """
        print(f"\n保存投影数据到: {self.output_dir}")
        
        if save_individual:
            # 创建子目录存储单独的投影帧
            frames_dir = os.path.join(self.output_dir, 'frames')
            os.makedirs(frames_dir, exist_ok=True)
            
            print(f"  保存单独投影帧...")
            for i in range(projections.shape[0]):
                frame_filename = os.path.join(frames_dir, f'projection_{i:04d}.raw')
                projections[i].astype(np.float32).tofile(frame_filename)
                
                if (i + 1) % 100 == 0:
                    print(f"    已保存 {i + 1}/{projections.shape[0]} 帧")
            
            print(f"  ✓ 已保存 {projections.shape[0]} 个投影帧到: {frames_dir}")
        
        if save_stack:
            # 保存完整投影栈
            stack_filename = os.path.join(self.output_dir, 'projections_stack.raw')
            projections.astype(np.float32).tofile(stack_filename)
            print(f"  ✓ 已保存投影栈到: {stack_filename}")
        
        # 保存角度信息
        angles_filename = os.path.join(self.output_dir, 'angles.txt')
        with open(angles_filename, 'w') as f:
            f.write(f"# 投影角度信息\n")
            f.write(f"# 总角度数: {len(angles)}\n")
            f.write(f"# 角度范围: {angles[0]:.6f} - {angles[-1]:.6f} 弧度\n")
            f.write(f"# 角度范围: {np.degrees(angles[0]):.2f}° - {np.degrees(angles[-1]):.2f}°\n")
            f.write(f"# 角度间隔: {(angles[1]-angles[0]):.6f} 弧度 ({np.degrees(angles[1]-angles[0]):.2f}°)\n")
            f.write(f"#\n")
            f.write(f"# 索引\t角度(弧度)\t角度(度)\n")
            for i, angle in enumerate(angles):
                f.write(f"{i}\t{angle:.6f}\t{np.degrees(angle):.2f}\n")
        
        print(f"  ✓ 已保存角度信息到: {angles_filename}")
        
        # 保存元数据
        meta_filename = os.path.join(self.output_dir, 'projection_info.txt')
        with open(meta_filename, 'w') as f:
            f.write(f"TIGRE投影数据信息\n")
            f.write(f"{'='*60}\n")
            f.write(f"源RAW文件: {self.raw_file}\n")
            f.write(f"体块形状: {self.volume_shape} (Z, Y, X)\n")
            f.write(f"数据类型: {self.dtype}\n")
            f.write(f"\n")
            f.write(f"投影参数:\n")
            f.write(f"  角度数量: {len(angles)}\n")
            f.write(f"  角度范围: {np.degrees(angles[0]):.1f}° - {np.degrees(angles[-1]):.1f}°\n")
            f.write(f"  探测器尺寸: {projections.shape[1]} × {projections.shape[2]}\n")
            f.write(f"  投影数据形状: {projections.shape}\n")
            f.write(f"  投影数值范围: [{projections.min():.4f}, {projections.max():.4f}]\n")
            f.write(f"\n")
            f.write(f"几何配置:\n")
            f.write(f"  DSD: {self.geo.DSD} mm\n")
            f.write(f"  DSO: {self.geo.DSO} mm\n")
            f.write(f"  探测器像素: {self.geo.nDetector}\n")
            f.write(f"  探测器像素大小: {self.geo.dDetector} mm\n")
            f.write(f"  体素数量: {self.geo.nVoxel}\n")
            f.write(f"  体素大小: {self.geo.dVoxel} mm\n")
        
        print(f"  ✓ 已保存元数据到: {meta_filename}")
    
    def visualize_sample_projections(self, projections: np.ndarray, 
                                    angles: np.ndarray, num_samples: int = 4):
        """
        可视化部分投影样例
        
        Args:
            projections: 投影数据
            angles: 角度数组
            num_samples: 显示的样例数量
        """
        print(f"\n生成可视化...")
        
        # 选择均匀分布的样例
        indices = np.linspace(0, len(angles)-1, num_samples, dtype=int)
        
        fig, axes = plt.subplots(1, num_samples, figsize=(4*num_samples, 4))
        
        if num_samples == 1:
            axes = [axes]
        
        for idx, ax in zip(indices, axes):
            im = ax.imshow(projections[idx], cmap='gray')
            ax.set_title(f'Angle: {np.degrees(angles[idx]):.1f}°\n(Index: {idx})')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        plt.tight_layout()
        
        # 保存图像
        viz_filename = os.path.join(self.output_dir, 'sample_projections.png')
        plt.savefig(viz_filename, dpi=150, bbox_inches='tight')
        print(f"  ✓ 已保存可视化到: {viz_filename}")
        
        plt.close()
    
    def run(self, num_angles: int = 360, 
            save_individual: bool = True,
            save_stack: bool = True,
            visualize: bool = True):
        """
        执行完整的投影模拟流程
        
        Args:
            num_angles: 投影角度数量
            save_individual: 是否保存单独的投影帧
            save_stack: 是否保存投影栈
            visualize: 是否生成可视化
        """
        print("="*60)
        print("TIGRE投影模拟工具")
        print("="*60)
        
        try:
            # 1. 加载体块
            volume = self.load_raw_volume()
            
            # 2. 生成投影
            projections, angles = self.generate_projections(volume, num_angles)
            
            # 3. 保存投影
            self.save_projections(projections, angles, save_individual, save_stack)
            
            # 4. 可视化
            if visualize:
                self.visualize_sample_projections(projections, angles)
            
            print("\n" + "="*60)
            print("投影模拟完成!")
            print("="*60)
            
        except Exception as e:
            print(f"\n错误: {e}")
            import traceback
            traceback.print_exc()
            raise


def parse_args():
    parser = argparse.ArgumentParser(
        description="TIGRE投影模拟工具 - 读取RAW体块并生成360个角度的模拟投影",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="输入RAW文件路径"
    )
    parser.add_argument(
        "-s", "--shape",
        nargs=3,
        type=int,
        required=True,
        metavar=('Z', 'Y', 'X'),
        help="体块形状 (Z Y X)"
    )
    parser.add_argument(
        "--dtype",
        choices=['float32', 'float64', 'int16', 'uint16', 'uint8', 'int32', 'uint32'],
        default='int16',
        help="RAW文件数据类型"
    )
    parser.add_argument(
        "-o", "--output",
        default="./projection_output",
        help="输出目录"
    )
    parser.add_argument(
        "-n", "--num-angles",
        type=int,
        default=360,
        help="投影角度数量"
    )
    parser.add_argument(
        "--no-individual",
        action="store_true",
        help="不保存单独的投影帧"
    )
    parser.add_argument(
        "--no-stack",
        action="store_true",
        help="不保存投影栈"
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="不生成可视化"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        return
    
    # 创建模拟器
    simulator = TigreProjectionSimulator(
        raw_file=args.input,
        volume_shape=tuple(args.shape),
        dtype=args.dtype,
        output_dir=args.output
    )
    
    # 执行模拟
    simulator.run(
        num_angles=args.num_angles,
        save_individual=not args.no_individual,
        save_stack=not args.no_stack,
        visualize=not args.no_viz
    )


if __name__ == "__main__":
    main()
