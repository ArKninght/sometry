#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CT数据组合工具
将分开的.raw文件重新组合成完整的多帧CT数据
"""

import numpy as np
import os
import glob
import yaml
from pathlib import Path
from typing import List, Tuple, Dict, Optional


class CTDataComposer:
    """CT数据组合器"""
    
    def __init__(self, config_file: str = "datainfo.yaml"):
        """
        初始化组合器
        
        参数:
            config_file: 配置文件路径
        """
        self.config = self._load_config(config_file)
        self._validate_config()
        
        # 数据类型映射
        self.dtype_map = {
            'uint8': np.uint8,
            'uint16': np.uint16,
            'uint32': np.uint32,
            'float32': np.float32,
            'float64': np.float64,
            'int16': np.int16,
            'int32': np.int32
        }
    
    def _load_config(self, config_file: str) -> Dict:
        """加载配置文件"""
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"配置文件不存在: {config_file}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _validate_config(self):
        """验证配置文件"""
        required_keys = ['input', 'output', 'data', 'compose']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"配置文件缺少必需的键: {key}")
        
        # 验证数据类型
        dtype = self.config['data'].get('dtype', 'float32')
        if dtype not in self.dtype_map:
            raise ValueError(f"不支持的数据类型: {dtype}")
    
    def _get_dtype(self) -> np.dtype:
        """获取numpy数据类型"""
        dtype_str = self.config['data'].get('dtype', 'float32')
        return self.dtype_map[dtype_str]
    
    def _get_volume_shape(self) -> Tuple[int, int, int]:
        """获取体块尺寸 (深度, 高度, 宽度)"""
        width = self.config['data']['width']
        height = self.config['data']['height']
        depth = self.config['data']['depth']
        return depth, height, width
    
    def _collect_files(self) -> Dict[str, List[str]]:
        """
        收集需要组合的文件
        
        返回:
            字典 {原始文件名: [frame1_path, frame2_path, ...]}
        """
        input_config = self.config['input']
        input_folder = input_config['folder']
        structure = input_config.get('structure', 'frames')
        num_frames = input_config['num_frames']
        file_pattern = input_config.get('file_pattern', '*.raw')
        
        file_groups = {}
        
        if structure == 'frames':
            # 按帧分文件夹的结构 (image_1, image_2, ...)
            for frame_idx in range(1, num_frames + 1):
                frame_folder = os.path.join(input_folder, f"image_{frame_idx}")
                
                if not os.path.exists(frame_folder):
                    print(f"警告: 文件夹不存在 - {frame_folder}")
                    continue
                
                # 获取该帧的所有文件
                frame_files = sorted(glob.glob(os.path.join(frame_folder, file_pattern)))
                
                for file_path in frame_files:
                    # 提取基础文件名（去除_frame_X部分）
                    basename = os.path.basename(file_path)
                    # 尝试提取原始文件名
                    base_name = basename.replace(f'_frame_{frame_idx}.raw', '')
                    
                    if base_name not in file_groups:
                        file_groups[base_name] = [None] * num_frames
                    
                    file_groups[base_name][frame_idx - 1] = file_path
        
        elif structure == 'files':
            # 所有文件在同一目录
            all_files = sorted(glob.glob(os.path.join(input_folder, file_pattern)))
            
            # 按文件名分组
            for file_path in all_files:
                basename = os.path.basename(file_path)
                # 提取帧号和基础名
                # 假设格式为: name_frame_X.raw
                parts = basename.rsplit('_frame_', 1)
                if len(parts) == 2:
                    base_name = parts[0]
                    frame_num = int(parts[1].replace('.raw', ''))
                    
                    if base_name not in file_groups:
                        file_groups[base_name] = [None] * num_frames
                    
                    if 0 < frame_num <= num_frames:
                        file_groups[base_name][frame_num - 1] = file_path
        
        # 过滤掉不完整的文件组
        complete_groups = {}
        for base_name, frame_files in file_groups.items():
            if None not in frame_files:
                complete_groups[base_name] = frame_files
            else:
                missing = [i+1 for i, f in enumerate(frame_files) if f is None]
                print(f"警告: 文件组 '{base_name}' 不完整，缺少帧: {missing}")
        
        return complete_groups
    
    def _read_volume(self, file_path: str) -> np.ndarray:
        """读取单个体块文件"""
        dtype = self._get_dtype()
        depth, height, width = self._get_volume_shape()
        
        data = np.fromfile(file_path, dtype=dtype)
        expected_size = depth * height * width
        
        if len(data) != expected_size:
            raise ValueError(
                f"文件大小不匹配: {file_path}\n"
                f"预期: {expected_size} 体素 ({width}×{height}×{depth}), "
                f"实际: {len(data)} 体素"
            )
        
        return data.reshape(depth, height, width)
    
    def compose_sequential(self, file_groups: Dict[str, List[str]]) -> None:
        """
        按顺序组合模式 - 将所有帧组合到单个文件
        
        参数:
            file_groups: 文件组字典
        """
        output_config = self.config['output']
        output_folder = output_config['folder']
        output_prefix = output_config.get('prefix', 'composed')
        overwrite = output_config.get('overwrite', False)
        
        # 创建输出目录
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"\n开始组合 {len(file_groups)} 组数据...")
        
        for idx, (base_name, frame_files) in enumerate(file_groups.items(), 1):
            output_file = os.path.join(output_folder, f"{output_prefix}_{base_name}.raw")
            
            # 检查文件是否存在
            if os.path.exists(output_file) and not overwrite:
                print(f"[{idx}/{len(file_groups)}] 跳过 (已存在): {os.path.basename(output_file)}")
                continue
            
            print(f"[{idx}/{len(file_groups)}] 处理: {base_name}")
            
            # 读取所有体块
            volumes = []
            for vol_idx, vol_file in enumerate(frame_files, 1):
                try:
                    volume_data = self._read_volume(vol_file)
                    volumes.append(volume_data)
                except Exception as e:
                    print(f"  错误: 读取体块 {vol_idx} 失败 - {e}")
                    break
            
            if len(volumes) != len(frame_files):
                print(f"  跳过此文件组（读取失败）")
                continue
            
            # 组合所有体块
            combined_data = np.stack(volumes, axis=0)
            
            # 保存
            combined_data.tofile(output_file)
            
            print(f"  ✓ 已保存: {os.path.basename(output_file)}")
            print(f"    形状: {combined_data.shape} (体块数×深度×高度×宽度)")
            print(f"    数据范围: [{combined_data.min():.2f}, {combined_data.max():.2f}]")
        
        print(f"\n组合完成! 输出目录: {output_folder}")
    
    def compose_separate(self, file_groups: Dict[str, List[str]]) -> None:
        """
        分离组合模式 - 每组帧生成独立文件（保持分离状态）
        
        参数:
            file_groups: 文件组字典
        """
        output_config = self.config['output']
        output_folder = output_config['folder']
        overwrite = output_config.get('overwrite', False)
        num_frames = self.config['input']['num_frames']
        
        # 为每一帧创建输出文件夹
        for frame_idx in range(1, num_frames + 1):
            frame_folder = os.path.join(output_folder, f"frame_{frame_idx}")
            os.makedirs(frame_folder, exist_ok=True)
        
        print(f"\n开始复制 {len(file_groups)} 组数据到新位置...")
        
        for idx, (base_name, frame_files) in enumerate(file_groups.items(), 1):
            print(f"[{idx}/{len(file_groups)}] 处理: {base_name}")
            
            for frame_idx, frame_file in enumerate(frame_files, 1):
                frame_folder = os.path.join(output_folder, f"frame_{frame_idx}")
                output_file = os.path.join(frame_folder, f"{base_name}.raw")
                
                if os.path.exists(output_file) and not overwrite:
                    continue
                
                # 读取并保存
                try:
                    volume_data = self._read_volume(frame_file)
                    volume_data.tofile(output_file)
                except Exception as e:
                    print(f"  错误: 处理体块 {frame_idx} 失败 - {e}")
        
        print(f"\n复制完成! 输出目录: {output_folder}")
    
    def compose(self) -> None:
        """执行组合操作"""
        # 收集文件
        print("收集文件...")
        file_groups = self._collect_files()
        
        if not file_groups:
            print("错误: 未找到可组合的文件")
            return
        
        print(f"找到 {len(file_groups)} 组完整数据")
        
        # 根据模式执行组合
        compose_mode = self.config['compose'].get('mode', 'sequential')
        
        if compose_mode == 'sequential':
            self.compose_sequential(file_groups)
        elif compose_mode == 'separate':
            self.compose_separate(file_groups)
        else:
            raise ValueError(f"不支持的组合模式: {compose_mode}")
    
    def verify_output(self) -> None:
        """验证输出结果"""
        output_config = self.config['output']
        output_folder = output_config['folder']
        compose_mode = self.config['compose'].get('mode', 'sequential')
        
        if not os.path.exists(output_folder):
            print(f"输出目录不存在: {output_folder}")
            return
        
        print("\n验证输出结果:")
        
        if compose_mode == 'sequential':
            output_files = glob.glob(os.path.join(output_folder, "*.raw"))
            
            if not output_files:
                print("✗ 未找到输出文件")
                return
            
            dtype = self._get_dtype()
            depth, height, width = self._get_volume_shape()
            num_volumes = self.config['input']['num_frames']
            voxels_per_volume = depth * height * width
            expected_size = voxels_per_volume * num_volumes
            
            for file_path in sorted(output_files):
                data = np.fromfile(file_path, dtype=dtype)
                
                if len(data) == expected_size:
                    data_4d = data.reshape(num_volumes, depth, height, width)
                    print(f"✓ {os.path.basename(file_path)}")
                    print(f"  形状: {data_4d.shape} (体块数×深度×高度×宽度)")
                    print(f"  数据范围: [{data.min():.2f}, {data.max():.2f}]")
                else:
                    print(f"✗ {os.path.basename(file_path)}")
                    print(f"  大小不匹配 (预期 {expected_size} 体素, 实际 {len(data)} 体素)")
        
        elif compose_mode == 'separate':
            num_frames = self.config['input']['num_frames']
            
            for frame_idx in range(1, num_frames + 1):
                frame_folder = os.path.join(output_folder, f"frame_{frame_idx}")
                
                if not os.path.exists(frame_folder):
                    print(f"✗ 文件夹不存在: frame_{frame_idx}")
                    continue
                
                frame_files = glob.glob(os.path.join(frame_folder, "*.raw"))
                print(f"✓ frame_{frame_idx}: {len(frame_files)} 个文件")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='CT数据组合工具')
    parser.add_argument(
        '-c', '--config',
        default='datainfo.yaml',
        help='配置文件路径 (默认: datainfo.yaml)'
    )
    parser.add_argument(
        '-v', '--verify',
        action='store_true',
        help='验证输出结果'
    )
    
    args = parser.parse_args()
    
    try:
        # 创建组合器
        composer = CTDataComposer(args.config)
        
        # 执行组合
        composer.compose()
        
        # 验证结果
        if args.verify:
            composer.verify_output()
    
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()