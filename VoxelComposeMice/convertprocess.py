#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reverse processing pipeline.
Reads a normalized/int16 NIFTI file produced by dcmcompose.py and restores it
back to HU space by undoing the normalization and HU adjustments recorded in
hu_adjustment_stats.yaml.
"""

import argparse
import os
from typing import Optional, Tuple

import nibabel as nib
import numpy as np
import yaml


class ReverseConversionPipeline:
    """Undo normalization and HU adjustments using saved statistics."""

    def __init__(self, stats_path: str, target_range: Tuple[float, float]):
        self.stats_path = stats_path
        self.target_range = target_range
        self.stats = self._load_stats(stats_path)
        self.adjusted_range = self._require_key("adjusted_range")
        self.adjustment_params = self.stats.get("adjustment_params")
        self.roi_coordinates = self.stats.get("roi_coordinates")
        self.original_shape = self.stats.get("original_volume_shape")

    def _require_key(self, key: str):
        if key not in self.stats:
            raise KeyError(f"统计文件 {self.stats_path} 中缺少 '{key}' 字段")
        return self.stats[key]

    @staticmethod
    def _load_stats(stats_path: str) -> dict:
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"统计文件不存在: {stats_path}")
        with open(stats_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not data:
            raise ValueError(f"{stats_path} 是空文件或解析失败")
        return data

    @staticmethod
    def load_nifti(nifti_path: str):
        if not os.path.exists(nifti_path):
            raise FileNotFoundError(f"NIFTI文件不存在: {nifti_path}")
        img = nib.load(nifti_path)
        data = img.get_fdata(dtype=np.float32)
        print(f"读取 {nifti_path}")
        print(f"  形状: {data.shape}")
        print(f"  数据范围: [{data.min():.2f}, {data.max():.2f}] (归一化空间)")
        return data, img

    def inverse_normalize(self, normalized_volume: np.ndarray) -> np.ndarray:
        """Map data from target_range back to the adjusted HU range."""
        t_min, t_max = self.target_range
        a_min = float(self.adjusted_range["min"])
        a_max = float(self.adjusted_range["max"])
        if abs(t_max - t_min) < 1e-6:
            raise ValueError("target_range 的上下限太接近，无法逆归一化")
        print("\n逆归一化...")
        print(f"  归一化目标范围: [{t_min:.2f}, {t_max:.2f}]")
        print(f"  调整后HU范围: [{a_min:.2f}, {a_max:.2f}]")
        scale = (a_max - a_min) / (t_max - t_min)
        restored = (normalized_volume - t_min) * scale + a_min
        print(f"  逆归一化后范围: [{restored.min():.2f}, {restored.max():.2f}] HU")
        return restored

    def inverse_hu_adjustment(self, adjusted_volume: np.ndarray) -> np.ndarray:
        """Recover HU values before boost/reduce operations.
        
        正向调整逻辑:
        - 原始值 > upper_boundary(6000) → +high_boost(5000)
        - 原始值 < lower_boundary(500) → -low_reduce(500)
        - lower_boundary ≤ 原始值 ≤ upper_boundary → 压缩到threshold(1000)
        
        逆向恢复逻辑:
        - 调整后 > upper_boundary → -high_boost
        - 调整后 < lower_boundary → +low_reduce
        - lower_boundary ≤ 调整后 < threshold → 恢复为threshold
        - threshold ≤ 调整后 ≤ upper_boundary → 线性映射到[threshold, 1250]
        """
        if not self.adjustment_params:
            print("YAML中缺少 adjustment_params，跳过HU逆调整")
            return adjusted_volume

        threshold = float(self.adjustment_params.get("high_threshold", 1000.0))
        high_boost = float(self.adjustment_params.get("high_boost", 0.0))
        low_reduce = float(self.adjustment_params.get("low_reduce", 0.0))
        
        # 计算原始数据的上下边界
        upper_boundary = threshold + high_boost  # 6000
        lower_boundary = threshold - low_reduce  # 500
        
        # 中间区域的映射上限
        mid_upper_target = 1250.0

        print("\n逆向恢复HU调整...")
        print(f"  阈值threshold: {threshold}")
        print(f"  上边界(原始): {upper_boundary}")
        print(f"  下边界(原始): {lower_boundary}")
        print(f"  高值增益: +{high_boost}")
        print(f"  低值减量: -{low_reduce}")
        print(f"  中间区域映射: [{threshold}, {upper_boundary}] → [{threshold}, {mid_upper_target}]")

        restored = adjusted_volume.copy()
        
        # 高值区域: 调整后 > upper_boundary → 减去增益
        high_mask = restored > upper_boundary
        # 低值区域: 调整后 < lower_boundary → 加回减量
        low_mask = restored < lower_boundary
        # 中间下区域: lower_boundary ≤ 调整后 < threshold → 恢复为threshold
        mid_low_mask = (restored >= lower_boundary) & (restored < threshold)
        # 中间上区域: threshold ≤ 调整后 ≤ upper_boundary → 线性映射
        mid_high_mask = (restored >= threshold) & (restored <= upper_boundary)

        # 高低区域处理
        restored[high_mask] -= high_boost
        restored[low_mask] += low_reduce
        
        # 中间下区域恢复为阈值
        restored[mid_low_mask] = threshold
        
        # 中间上区域线性映射: [threshold, upper_boundary] → [threshold, 1250]
        if mid_high_mask.any():
            mid_high_values = restored[mid_high_mask]
            # 线性映射公式: (value - threshold) / (upper_boundary - threshold) * (1250 - threshold) + threshold
            scale = (mid_upper_target - threshold) / (upper_boundary - threshold)
            restored[mid_high_mask] = (mid_high_values - threshold) * scale + threshold
        
        print(f"  高值像素数: {high_mask.sum()} (恢复: 值 - {high_boost})")
        print(f"  低值像素数: {low_mask.sum()} (恢复: 值 + {low_reduce})")
        print(f"  中间下区像素数: {mid_low_mask.sum()} (恢复为: {threshold})")
        print(f"  中间上区像素数: {mid_high_mask.sum()} (映射到: [{threshold}, {mid_upper_target}])")
        print(f"  逆调整后范围: [{restored.min():.2f}, {restored.max():.2f}] HU")
        return restored

    def restore_full_volume(self, cropped: np.ndarray) -> np.ndarray:
        """Embed the cropped ROI back into an empty volume with original shape."""
        if not self.roi_coordinates or not self.original_shape:
            raise ValueError("统计文件中缺少 ROI/原始尺寸信息，无法还原到原尺寸")

        orig_shape = (
            int(self.original_shape["z"]),
            int(self.original_shape["y"]),
            int(self.original_shape["x"]),
        )
        print("\n根据ROI信息还原到原始尺寸...")
        print(f"  原始形状: {orig_shape}")
        print(f"  当前形状: {cropped.shape}")
        full_volume = np.zeros(orig_shape, dtype=cropped.dtype)

        z_min = int(self.roi_coordinates["z"]["min"])
        y_min = int(self.roi_coordinates["y"]["min"])
        x_min = int(self.roi_coordinates["x"]["min"])

        z_max = z_min + cropped.shape[0]
        y_max = y_min + cropped.shape[1]
        x_max = x_min + cropped.shape[2]

        full_volume[z_min:z_max, y_min:y_max, x_min:x_max] = cropped
        print("  已将ROI数据放回原体积，其余区域填0")
        return full_volume

    @staticmethod
    def save_nifti(volume: np.ndarray, reference_img: Optional[nib.Nifti1Image], output_path: str):
        """Persist restored volume as float32 NIFTI."""
        affine = reference_img.affine if reference_img is not None else np.eye(4)
        header = reference_img.header.copy() if reference_img is not None else None
        if header is not None:
            header.set_data_dtype(np.float32)

        output_img = nib.Nifti1Image(volume.astype(np.float32), affine, header)
        nib.save(output_img, output_path)
        print(f"\n已保存恢复后的NIFTI: {output_path}")
        print(f"  形状: {volume.shape}")
        print(f"  数据范围: [{volume.min():.2f}, {volume.max():.2f}] HU")

    def process(self, input_nifti: str, output_nifti: str, restore_full_volume: bool = False):
        data, img = self.load_nifti(input_nifti)
        denormalized = self.inverse_normalize(data)
        restored_hu = self.inverse_hu_adjustment(denormalized)
        if restore_full_volume:
            restored_hu = self.restore_full_volume(restored_hu)
        self.save_nifti(restored_hu, img if not restore_full_volume else None, output_nifti)


def parse_args():
    parser = argparse.ArgumentParser(
        description="逆向恢复 dcmcompose.py 导出的NIFTI文件",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input", required=True, help="归一化/int16后的NIFTI文件")
    parser.add_argument("-o", "--output", required=True, help="输出恢复后的NIFTI路径")
    parser.add_argument(
        "-s",
        "--stats",
        default="/home/pby/Documents/someTry/VoxelComposeMice/nifti_output/hu_adjustment_stats.yaml",
        help="hu_adjustment_stats.yaml 路径",
    )
    parser.add_argument(
        "-t",
        "--target-range",
        nargs=2,
        type=float,
        default=(-1000.0, 1000.0),
        metavar=("MIN", "MAX"),
        help="正向归一化时使用的目标范围",
    )
    parser.add_argument(
        "--restore-full-volume",
        action="store_true",
        help="使用ROI信息将恢复后的数据放回原始体积尺寸",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pipeline = ReverseConversionPipeline(args.stats, tuple(args.target_range))
    pipeline.process(args.input, args.output, restore_full_volume=args.restore_full_volume)


if __name__ == "__main__":
    main()
