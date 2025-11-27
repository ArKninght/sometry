#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NIFTI Volume Upsampling Tool
Upsample and resize 3D NIFTI volumes from 128x128x128 to 512x512x256 (or custom sizes)
"""

import argparse
import os
from typing import Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from PIL import Image


class VolumeUpsampler:
    """Volume upsampling and resizing utility"""

    def __init__(self, target_shape: Tuple[int, int, int] = (512, 512, 256)):
        """
        Args:
            target_shape: Desired output volume size (x, y, z)
        """
        self.target_shape = target_shape

    def load_nifti(self, nifti_file: str):
        """Load nifti data and metadata"""
        if not os.path.exists(nifti_file):
            raise FileNotFoundError(f"File not found: {nifti_file}")

        print(f"Loading: {nifti_file}")
        nifti_img = nib.load(nifti_file)
        volume = nifti_img.get_fdata()

        if volume.ndim == 4:
            print(f"  4D detected, using first timepoint out of {volume.shape[3]}")
            volume = volume[..., 0]

        print(f"  Shape: {volume.shape}")
        print(f"  Dtype: {volume.dtype}")
        print(f"  Range: [{volume.min():.2f}, {volume.max():.2f}]")

        return volume, nifti_img

    def _upsample_lanczos3(self, volume: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
        """
        使用PIL的LANCZOS快速重采样3D体块
        
        Args:
            volume: 输入3D数组
            target_shape: 目标形状 (x, y, z)
        
        Returns:
            重采样后的3D数组
        """
        print("  使用Lanczos-3插值 (基于PIL优化)")
        
        # 归一化到0-255范围以使用PIL
        vmin, vmax = volume.min(), volume.max()
        if vmax - vmin > 1e-6:
            volume_normalized = ((volume - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        else:
            volume_normalized = np.zeros(volume.shape, dtype=np.uint8)
        
        # 第1步: 重采样Z轴 (逐切片处理XY平面)
        print(f"  步骤1/2: 重采样Z轴 {volume.shape[2]} -> {target_shape[2]}")
        temp_z = np.zeros((volume.shape[0], volume.shape[1], target_shape[2]), dtype=np.uint8)
        
        for i in range(volume.shape[0]):
            # 将YZ平面转置为2D图像
            slice_2d = volume_normalized[i, :, :].T  # shape: (Z, Y)
            # 使用PIL重采样 (自动推断为'L'模式)
            img = Image.fromarray(slice_2d)
            img_resized = img.resize((volume.shape[1], target_shape[2]), Image.LANCZOS)
            temp_z[i, :, :] = np.array(img_resized).T
        
        # 第2步: 重采样XY平面
        print(f"  步骤2/2: 重采样XY平面 ({volume.shape[0]}, {volume.shape[1]}) -> ({target_shape[0]}, {target_shape[1]})")
        result_normalized = np.zeros(target_shape, dtype=np.uint8)
        
        for k in range(target_shape[2]):
            # XY切片
            slice_2d = temp_z[:, :, k]
            # 使用PIL重采样 (自动推断为'L'模式)
            img = Image.fromarray(slice_2d)
            img_resized = img.resize((target_shape[1], target_shape[0]), Image.LANCZOS)
            result_normalized[:, :, k] = np.array(img_resized)
        
        # 恢复到原始数值范围
        result = result_normalized.astype(np.float32) / 255.0 * (vmax - vmin) + vmin
        result = result.astype(volume.dtype)
        
        print(f"  Lanczos-3重采样完成: {result.shape}")
        return result

    def upsample_volume(self, volume: np.ndarray, method: str = "cubic") -> np.ndarray:
        """Upsample to target shape with interpolation method"""
        original_shape = volume.shape
        zoom_factors = [
            self.target_shape[0] / original_shape[0],
            self.target_shape[1] / original_shape[1],
            self.target_shape[2] / original_shape[2],
        ]

        print("\nUpsampling")
        print(f"  Original shape: {original_shape}")
        print(f"  Target shape: {self.target_shape}")
        print(
            f"  Zoom factors: [{zoom_factors[0]:.4f}, "
            f"{zoom_factors[1]:.4f}, {zoom_factors[2]:.4f}]"
        )

        # Handle Lanczos-3 separately
        if method == "lanczos3":
            upsampled = self._upsample_lanczos3(volume, self.target_shape)
        else:
            method_order = {"nearest": 0, "linear": 1, "cubic": 3}
            if method not in method_order:
                raise ValueError(f"Unsupported method {method}. Choose from {list(method_order.keys())} or 'lanczos3'")

            upsampled = zoom(volume, zoom_factors, order=method_order[method], mode="nearest")

        if upsampled.shape != self.target_shape:
            print(f"  Adjusting shape from {upsampled.shape} to {self.target_shape}")
            result = np.zeros(self.target_shape, dtype=upsampled.dtype)
            copy_slices = []
            for axis in range(3):
                src = upsampled.shape[axis]
                tgt = self.target_shape[axis]
                start_src = max(0, (src - tgt) // 2)
                end_src = start_src + min(src, tgt)
                start_tgt = max(0, (tgt - src) // 2)
                end_tgt = start_tgt + min(src, tgt)
                copy_slices.append((slice(start_src, end_src), slice(start_tgt, end_tgt)))

            result[
                copy_slices[0][1], copy_slices[1][1], copy_slices[2][1]
            ] = upsampled[copy_slices[0][0], copy_slices[1][0], copy_slices[2][0]]
            upsampled = result

        print(f"  Upsampled shape: {upsampled.shape}")
        print(f"  Range: [{upsampled.min():.2f}, {upsampled.max():.2f}]")
        return upsampled

    def save_nifti(self, volume: np.ndarray, output_file: str, original_img=None, use_int16: bool = True):
        """Save upsampled data to nifti"""
        print(f"\nSaving to: {output_file}")
        if use_int16:
            print("  Casting to int16")
            print(f"  Original dtype: {volume.dtype}")
            print(f"  Original range: [{volume.min():.2f}, {volume.max():.2f}]")
            volume = np.clip(volume, np.iinfo(np.int16).min, np.iinfo(np.int16).max).astype(np.int16)

        if original_img is not None:
            original_shape = original_img.shape[:3]
            scale_factors = [
                original_shape[0] / self.target_shape[0],
                original_shape[1] / self.target_shape[1],
                original_shape[2] / self.target_shape[2],
            ]
            affine = original_img.affine.copy()
            affine[0, 0] *= scale_factors[0]
            affine[1, 1] *= scale_factors[1]
            affine[2, 2] *= scale_factors[2]
        else:
            affine = np.eye(4)

        nifti_img = nib.Nifti1Image(volume, affine)
        nib.save(nifti_img, output_file)

        print("  Saved successfully")
        print(f"  Shape: {volume.shape}")
        print(f"  Dtype: {volume.dtype}")
        print(f"  Size: {volume.nbytes / (1024 * 1024):.2f} MB")

    def process(self, input_file: str, output_file: str, method: str):
        """Run upsampling pipeline"""
        try:
            volume, original_img = self.load_nifti(input_file)
            upsampled = self.upsample_volume(volume, method)
            self.save_nifti(upsampled, output_file, original_img)
            print("\nUpsampling completed successfully!")
        except Exception as exc:
            print(f"\nError: {exc}")
            import traceback

            traceback.print_exc()
            raise


def parse_args():
    parser = argparse.ArgumentParser(
        description="NIFTI Volume Upsampling Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python upsampling.py -i input_128.nii.gz -o up_512.nii.gz
  python upsampling.py -i input.nii.gz -o upsampled.nii.gz -m linear
  python upsampling.py -i input.nii.gz -o custom.nii.gz -s 512 512 256
        """,
    )
    parser.add_argument("-i", "--input", required=True, help="Input NIFTI file path")
    parser.add_argument("-o", "--output", required=True, help="Output NIFTI file path")
    parser.add_argument(
        "-s",
        "--size",
        nargs=3,
        type=int,
        default=[512, 512, 256],
        help="Target size (x y z), default: 512 512 256",
    )
    parser.add_argument(
        "-m",
        "--method",
        choices=["nearest", "linear", "cubic", "lanczos3"],
        default="lanczos3",
        help="Interpolation method (default: cubic)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    upsampler = VolumeUpsampler(tuple(args.size))
    upsampler.process(args.input, args.output, args.method)


if __name__ == "__main__":
    main()
