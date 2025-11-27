#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4DCT运动投影混合工具
从多个phase的投影数据中抽取对应角度,组合成模拟呼吸运动的360角度投影序列
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt


class MotionProjectionMixer:
    """运动投影混合器"""
    
    def __init__(self, phase_folders: List[str], output_folder: str = "./mixed_projections"):
        """
        初始化混合器
        
        Args:
            phase_folders: 各个phase的投影文件夹路径列表
            output_folder: 输出混合投影的文件夹
        """
        self.phase_folders = phase_folders
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)
        
        self.num_phases = len(phase_folders)
        print(f"初始化运动投影混合器")
        print(f"  Phase数量: {self.num_phases}")
        
    def load_projection_info(self, phase_folder: str) -> Tuple[int, Tuple[int, int]]:
        """
        加载投影信息
        
        Args:
            phase_folder: phase文件夹路径
        
        Returns:
            (投影数量, 探测器尺寸(rows, cols))
        """
        # 读取投影信息文件
        info_file = os.path.join(phase_folder, 'projection_info.txt')
        if not os.path.exists(info_file):
            raise FileNotFoundError(f"投影信息文件不存在: {info_file}")
        
        num_projections = 0
        detector_size = None
        
        with open(info_file, 'r', encoding='utf-8') as f:
            for line in f:
                if '角度数量:' in line:
                    num_projections = int(line.split(':')[1].strip())
                elif '探测器尺寸:' in line:
                    # 格式: "探测器尺寸: 256 × 256"
                    sizes = line.split(':')[1].strip().split('×')
                    detector_size = (int(sizes[0].strip()), int(sizes[1].strip()))
        
        if num_projections == 0 or detector_size is None:
            raise ValueError(f"无法从 {info_file} 中读取投影信息")
        
        return num_projections, detector_size
    
    def create_motion_pattern(self, num_angles: int, motion_type: str = 'sinusoidal') -> np.ndarray:
        """
        创建运动模式 - 每个角度对应一个phase
        
        Args:
            num_angles: 投影角度数量
            motion_type: 运动类型
                - 'sinusoidal': 正弦运动(模拟呼吸)
                - 'linear': 线性循环
                - 'sawtooth': 锯齿波
                - 'triangular': 三角波/往返模式(0→max→0循环)
                - 'breathing_with_rest': 呼吸帧与静息帧混合(每次呼吸帧之间插入1-2帧phase0静息帧)
        
        Returns:
            phase索引数组 (num_angles,)
        """
        print(f"\n创建运动模式: {motion_type}")
        
        if motion_type == 'sinusoidal':
            # 正弦运动 - 模拟呼吸周期
            # 0 -> max -> 0 -> max -> ... (平滑过渡)
            t = np.linspace(0, 4*np.pi, num_angles)  # 2个完整呼吸周期
            phase_continuous = (np.sin(t) + 1) / 2 * (self.num_phases - 1)
            
        elif motion_type == 'linear':
            # 线性循环 - 均匀遍历各phase
            phase_continuous = np.linspace(0, self.num_phases - 1, num_angles) % self.num_phases
            
        elif motion_type == 'sawtooth':
            # 锯齿波 - 快速返回
            cycles = 2  # 2个周期
            phase_continuous = (np.arange(num_angles) % (num_angles // cycles)) / (num_angles // cycles) * (self.num_phases - 1)
            
        elif motion_type == 'triangular':
            # 三角波/往返模式 - 0→1→2→3→2→1→0→1→2→3→...
            # 一个完整周期长度: 2*(num_phases-1) 个角度
            # 例如: 5个phase(0-4) → 周期长度=8 (0,1,2,3,4,3,2,1)
            phase_indices = []
            cycle_length = 2 * (self.num_phases - 1)
            
            for i in range(num_angles):
                position_in_cycle = i % cycle_length
                
                if position_in_cycle < self.num_phases:
                    # 上升阶段: 0→1→2→3→4
                    phase_idx = position_in_cycle
                else:
                    # 下降阶段: 3→2→1 (不包括0,因为下一轮从0开始)
                    phase_idx = cycle_length - position_in_cycle
                
                phase_indices.append(phase_idx)
            
            phase_continuous = np.array(phase_indices, dtype=float)
            
        elif motion_type == 'breathing_with_rest':
            # 呼吸帧与静息帧混合模式
            # 模式: 2-5帧phase0静息帧 -> 1帧呼吸帧(phase1-4随机) -> 2-5帧phase0 -> 1帧呼吸帧 -> ...
            print(f"  使用呼吸+静息混合模式")
            print(f"  - 静息帧: phase 0 (每次2-5帧)")
            print(f"  - 呼吸帧: phase 1-{self.num_phases-1} (随机,每次1帧)")
            
            phase_indices = []
            i = 0
            # np.random.seed(42)  # 设置随机种子以便复现
            np.random.seed(79)

            while i < num_angles:
                # 1. 添加2-5帧静息相位 (phase0)
                num_rest_frames = np.random.randint(2, 6)  # 随机2-5帧
                for _ in range(num_rest_frames):
                    if i >= num_angles:
                        break
                    phase_indices.append(0)  # phase0是静息帧
                    i += 1
                
                if i >= num_angles:
                    break
                
                # 2. 添加一帧呼吸相位 (从phase1-4随机选择)
                breathing_phase = np.random.randint(1, self.num_phases)
                phase_indices.append(breathing_phase)
                i += 1
            
            phase_continuous = np.array(phase_indices, dtype=float)
            
        else:
            raise ValueError(f"不支持的运动类型: {motion_type}")
        
        # 转换为整数phase索引
        phase_indices = np.round(phase_continuous).astype(int)
        phase_indices = np.clip(phase_indices, 0, self.num_phases - 1)
        
        # 统计每个phase的使用次数
        print(f"  运动模式统计:")
        for i in range(self.num_phases):
            count = np.sum(phase_indices == i)
            percentage = count / num_angles * 100
            print(f"    Phase {i}: {count} 次 ({percentage:.1f}%)")
        
        return phase_indices
    
    def load_projection_frame(self, phase_folder: str, angle_idx: int, 
                             detector_size: Tuple[int, int]) -> np.ndarray:
        """
        加载指定phase和角度的投影帧
        
        Args:
            phase_folder: phase文件夹路径
            angle_idx: 角度索引
            detector_size: 探测器尺寸
        
        Returns:
            投影数据 (rows, cols)
        """
        frame_file = os.path.join(phase_folder, 'frames', f'projection_{angle_idx:04d}.raw')
        
        if not os.path.exists(frame_file):
            raise FileNotFoundError(f"投影帧文件不存在: {frame_file}")
        
        # 读取RAW文件
        projection = np.fromfile(frame_file, dtype=np.float32)
        projection = projection.reshape(detector_size)
        
        return projection
    
    def mix_projections(self, num_angles: int = 360, motion_type: str = 'sinusoidal') -> Tuple[np.ndarray, np.ndarray]:
        """
        混合多个phase的投影数据
        
        Args:
            num_angles: 输出投影角度数量
            motion_type: 运动模式类型
        
        Returns:
            (混合投影数组, phase索引数组)
        """
        print("\n混合投影数据...")
        
        # 1. 加载第一个phase的投影信息
        phase0_folder = self.phase_folders[0]
        num_proj_per_phase, detector_size = self.load_projection_info(phase0_folder)
        
        print(f"  每个phase投影数: {num_proj_per_phase}")
        print(f"  探测器尺寸: {detector_size}")
        print(f"  输出角度数: {num_angles}")
        
        # 2. 创建运动模式
        phase_indices = self.create_motion_pattern(num_angles, motion_type)
        
        # 3. 初始化输出数组
        mixed_projections = np.zeros((num_angles, detector_size[0], detector_size[1]), dtype=np.float32)
        
        # 4. 混合投影
        print(f"\n  正在混合投影...")
        for angle_idx in range(num_angles):
            # 确定使用哪个phase
            phase_idx = phase_indices[angle_idx]
            phase_folder = self.phase_folders[phase_idx]
            
            # 计算在该phase中对应的角度索引
            # 假设所有phase的角度范围相同,均匀分布
            source_angle_idx = int(angle_idx * num_proj_per_phase / num_angles)
            source_angle_idx = min(source_angle_idx, num_proj_per_phase - 1)
            
            # 加载投影帧
            try:
                projection = self.load_projection_frame(phase_folder, source_angle_idx, detector_size)
                mixed_projections[angle_idx] = projection
            except FileNotFoundError as e:
                print(f"    警告: {e}")
                # 如果找不到文件,使用前一帧或零填充
                if angle_idx > 0:
                    mixed_projections[angle_idx] = mixed_projections[angle_idx - 1]
            
            if (angle_idx + 1) % 50 == 0:
                print(f"    已处理 {angle_idx + 1}/{num_angles} 个角度")
        
        print(f"  混合投影完成")
        print(f"  输出形状: {mixed_projections.shape}")
        print(f"  数值范围: [{mixed_projections.min():.4f}, {mixed_projections.max():.4f}]")
        
        return mixed_projections, phase_indices
    
    def save_mixed_projections(self, projections: np.ndarray, phase_indices: np.ndarray,
                              save_individual: bool = True, save_stack: bool = True):
        """
        保存混合投影数据
        
        Args:
            projections: 混合投影数组
            phase_indices: 每个角度对应的phase索引
            save_individual: 是否保存单独的投影帧
            save_stack: 是否保存投影栈
        """
        print(f"\n保存混合投影到: {self.output_folder}")
        
        if save_individual:
            # 创建子目录
            frames_dir = os.path.join(self.output_folder, 'frames')
            os.makedirs(frames_dir, exist_ok=True)
            
            print(f"  保存单独投影帧...")
            for i in range(projections.shape[0]):
                frame_filename = os.path.join(frames_dir, f'mixed_projection_{i:04d}.raw')
                projections[i].astype(np.float32).tofile(frame_filename)
                
                if (i + 1) % 100 == 0:
                    print(f"    已保存 {i + 1}/{projections.shape[0]} 帧")
            
            print(f"  ✓ 已保存 {projections.shape[0]} 个投影帧到: {frames_dir}")
        
        if save_stack:
            # 保存完整投影栈
            stack_filename = os.path.join(self.output_folder, 'mixed_projections_stack.raw')
            projections.astype(np.float32).tofile(stack_filename)
            print(f"  ✓ 已保存投影栈到: {stack_filename}")
        
        # 保存phase信息
        phase_filename = os.path.join(self.output_folder, 'phase_mapping.txt')
        with open(phase_filename, 'w') as f:
            f.write(f"# 混合投影phase映射信息\n")
            f.write(f"# 总角度数: {len(phase_indices)}\n")
            f.write(f"# Phase数量: {self.num_phases}\n")
            f.write(f"#\n")
            f.write(f"# 角度索引\tPhase索引\n")
            for i, phase_idx in enumerate(phase_indices):
                f.write(f"{i}\t{phase_idx}\n")
        
        print(f"  ✓ 已保存phase映射到: {phase_filename}")
        
        # 保存元数据
        meta_filename = os.path.join(self.output_folder, 'mixed_projection_info.txt')
        with open(meta_filename, 'w') as f:
            f.write(f"混合投影数据信息\n")
            f.write(f"{'='*60}\n")
            f.write(f"源Phase文件夹:\n")
            for i, folder in enumerate(self.phase_folders):
                f.write(f"  Phase {i}: {folder}\n")
            f.write(f"\n")
            f.write(f"投影参数:\n")
            f.write(f"  角度数量: {len(phase_indices)}\n")
            f.write(f"  探测器尺寸: {projections.shape[1]} × {projections.shape[2]}\n")
            f.write(f"  投影数据形状: {projections.shape}\n")
            f.write(f"  数值范围: [{projections.min():.4f}, {projections.max():.4f}]\n")
            f.write(f"\n")
            f.write(f"Phase使用统计:\n")
            for i in range(self.num_phases):
                count = np.sum(phase_indices == i)
                percentage = count / len(phase_indices) * 100
                f.write(f"  Phase {i}: {count} 次 ({percentage:.1f}%)\n")
        
        print(f"  ✓ 已保存元数据到: {meta_filename}")
    
    def visualize_motion_pattern(self, phase_indices: np.ndarray, projections: np.ndarray):
        """
        可视化运动模式和样例投影
        
        Args:
            phase_indices: phase索引数组
            projections: 投影数组
        """
        print(f"\n生成可视化...")
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # 上图: 运动模式曲线
        ax1 = axes[0]
        angles = np.linspace(0, 360, len(phase_indices))
        ax1.plot(angles, phase_indices, 'b-', linewidth=1.5)
        ax1.set_xlabel('投影角度 (度)', fontsize=12)
        ax1.set_ylabel('Phase索引', fontsize=12)
        ax1.set_title('运动模式 - Phase vs 角度', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 360)
        ax1.set_ylim(-0.5, self.num_phases - 0.5)
        
        # 添加phase标记
        for i in range(self.num_phases):
            ax1.axhline(y=i, color='gray', linestyle='--', alpha=0.3)
            ax1.text(5, i, f'Phase {i}', fontsize=10, verticalalignment='center')
        
        # 下图: 样例投影 (选择4个均匀分布的角度)
        num_samples = min(4, len(projections))
        indices = np.linspace(0, len(projections)-1, num_samples, dtype=int)
        
        ax2 = axes[1]
        ax2.axis('off')
        
        # 创建子图显示样例投影
        for i, idx in enumerate(indices):
            sub_ax = fig.add_subplot(2, num_samples, num_samples + i + 1)
            im = sub_ax.imshow(projections[idx], cmap='gray')
            sub_ax.set_title(f'角度: {angles[idx]:.1f}°\nPhase: {phase_indices[idx]}', 
                           fontsize=10)
            sub_ax.axis('off')
            plt.colorbar(im, ax=sub_ax, fraction=0.046)
        
        plt.tight_layout()
        
        # 保存图像
        viz_filename = os.path.join(self.output_folder, 'motion_pattern_visualization.png')
        plt.savefig(viz_filename, dpi=150, bbox_inches='tight')
        print(f"  ✓ 已保存可视化到: {viz_filename}")
        
        plt.close()
    
    def run(self, num_angles: int = 360, 
            motion_type: str = 'sinusoidal',
            save_individual: bool = True,
            save_stack: bool = True,
            visualize: bool = True):
        """
        执行完整的混合流程
        
        Args:
            num_angles: 输出投影角度数量
            motion_type: 运动模式类型
            save_individual: 是否保存单独投影帧
            save_stack: 是否保存投影栈
            visualize: 是否生成可视化
        """
        print("="*60)
        print("4DCT运动投影混合工具")
        print("="*60)
        
        try:
            # 1. 混合投影
            projections, phase_indices = self.mix_projections(num_angles, motion_type)
            
            # 2. 保存投影
            self.save_mixed_projections(projections, phase_indices, save_individual, save_stack)
            
            # 3. 可视化
            if visualize:
                self.visualize_motion_pattern(phase_indices, projections)
            
            print("\n" + "="*60)
            print("混合完成!")
            print("="*60)
            
        except Exception as e:
            print(f"\n错误: {e}")
            import traceback
            traceback.print_exc()
            raise


def parse_args():
    parser = argparse.ArgumentParser(
        description="4DCT运动投影混合工具 - 从多个phase投影中组合运动序列",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "-p", "--phases",
        nargs='+',
        required=True,
        help="各个phase的投影文件夹路径 (按phase顺序)"
    )
    parser.add_argument(
        "-o", "--output",
        default="./mixed_projections",
        help="输出文件夹路径"
    )
    parser.add_argument(
        "-n", "--num-angles",
        type=int,
        default=360,
        help="输出投影角度数量"
    )
    parser.add_argument(
        "-m", "--motion-type",
        choices=['sinusoidal', 'linear', 'sawtooth', 'triangular', 'breathing_with_rest'],
        default='sinusoidal',
        help="运动模式类型"
    )
    parser.add_argument(
        "--no-individual",
        action="store_true",
        help="不保存单独投影帧"
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
    
    # 检查phase文件夹
    for phase_folder in args.phases:
        if not os.path.exists(phase_folder):
            print(f"错误: Phase文件夹不存在: {phase_folder}")
            return
    
    # 创建混合器
    mixer = MotionProjectionMixer(
        phase_folders=args.phases,
        output_folder=args.output
    )
    
    # 执行混合
    mixer.run(
        num_angles=args.num_angles,
        motion_type=args.motion_type,
        save_individual=not args.no_individual,
        save_stack=not args.no_stack,
        visualize=not args.no_viz
    )


if __name__ == "__main__":
    main()