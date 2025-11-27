#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DICOM to NIFTI 转换工具
将一组CT重建的DICOM文件组合成NIFTI格式
"""

import numpy as np
import os
import glob
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import pydicom
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, SpanSelector
from matplotlib.patches import Rectangle
import yaml


class DicomToNiftiConverter:
    """DICOM到NIFTI转换器"""
    
    def __init__(self, dicom_folder: str, output_folder: str = "./nifti_output"):
        """
        初始化转换器
        
        参数:
            dicom_folder: DICOM文件所在文件夹
            output_folder: 输出NIFTI文件的文件夹
        """
        self.dicom_folder = dicom_folder
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)
    
    def collect_dicom_files(self) -> List[str]:
        """
        收集DICOM文件
        
        返回:
            排序后的DICOM文件路径列表
        """
        # 查找所有.dcm文件
        dicom_files = glob.glob(os.path.join(self.dicom_folder, "*.dcm"))
        
        if not dicom_files:
            # 尝试不带扩展名的文件
            all_files = glob.glob(os.path.join(self.dicom_folder, "*"))
            dicom_files = []
            for f in all_files:
                if os.path.isfile(f):
                    try:
                        pydicom.dcmread(f, stop_before_pixels=True)
                        dicom_files.append(f)
                    except:
                        continue
        
        if not dicom_files:
            raise ValueError(f"在 {self.dicom_folder} 中未找到DICOM文件")
        
        print(f"找到 {len(dicom_files)} 个DICOM文件")
        
        # 读取并按位置排序
        dicom_data = []
        for dcm_file in dicom_files:
            try:
                dcm = pydicom.dcmread(dcm_file, stop_before_pixels=True)
                # 获取切片位置
                if hasattr(dcm, 'ImagePositionPatient'):
                    z_pos = float(dcm.ImagePositionPatient[2])
                elif hasattr(dcm, 'SliceLocation'):
                    z_pos = float(dcm.SliceLocation)
                elif hasattr(dcm, 'InstanceNumber'):
                    z_pos = float(dcm.InstanceNumber)
                else:
                    z_pos = 0
                
                dicom_data.append((z_pos, dcm_file))
            except Exception as e:
                print(f"警告: 无法读取 {dcm_file}: {e}")
                continue
        
        # 按z位置排序
        dicom_data.sort(key=lambda x: x[0])
        sorted_files = [f for _, f in dicom_data]
        
        print(f"成功排序 {len(sorted_files)} 个DICOM文件")
        
        return sorted_files
    
    def read_dicom_series(self, dicom_files: List[str]) -> Tuple[np.ndarray, dict]:
        """
        读取DICOM序列
        
        参数:
            dicom_files: DICOM文件路径列表
            
        返回:
            (3D numpy数组, 元数据字典)
        """
        print("\n读取DICOM序列...")
        
        # 读取第一个文件获取尺寸信息
        first_dcm = pydicom.dcmread(dicom_files[0])
        rows = int(first_dcm.Rows)
        cols = int(first_dcm.Columns)
        num_slices = len(dicom_files)
        
        print(f"图像尺寸: {cols} × {rows} × {num_slices}")
        
        # 初始化3D数组
        volume = np.zeros((num_slices, rows, cols), dtype=np.float32)
        
        # 读取所有切片
        for i, dcm_file in enumerate(dicom_files):
            try:
                dcm = pydicom.dcmread(dcm_file)
                pixel_array = dcm.pixel_array.astype(np.float32)
                
                # 应用Rescale Slope和Intercept (将像素值转换为HU值)
                if hasattr(dcm, 'RescaleSlope') and hasattr(dcm, 'RescaleIntercept'):
                    slope = float(dcm.RescaleSlope)
                    intercept = float(dcm.RescaleIntercept)
                    pixel_array = pixel_array * slope + intercept
                
                volume[i, :, :] = pixel_array
                
                if (i + 1) % 50 == 0:
                    print(f"  已读取 {i + 1}/{num_slices} 个切片")
                    
            except Exception as e:
                print(f"警告: 读取切片 {i} 失败: {e}")
                continue
        
        print(f"完成读取，数据范围: [{volume.min():.2f}, {volume.max():.2f}] HU")
        
        # 提取元数据
        metadata = self._extract_metadata(first_dcm)
        
        return volume, metadata
    
    def _extract_metadata(self, dcm: pydicom.Dataset) -> dict:
        """提取DICOM元数据"""
        metadata = {}
        
        # 体素间距 (mm)
        if hasattr(dcm, 'PixelSpacing'):
            pixel_spacing = [float(x) for x in dcm.PixelSpacing]
            metadata['pixel_spacing'] = pixel_spacing
        else:
            metadata['pixel_spacing'] = [1.0, 1.0]
        
        # 切片厚度 (mm)
        if hasattr(dcm, 'SliceThickness'):
            metadata['slice_thickness'] = float(dcm.SliceThickness)
        else:
            metadata['slice_thickness'] = 1.0
        
        # 患者信息
        if hasattr(dcm, 'PatientName'):
            metadata['patient_name'] = str(dcm.PatientName)
        if hasattr(dcm, 'PatientID'):
            metadata['patient_id'] = str(dcm.PatientID)
        
        # 扫描参数
        if hasattr(dcm, 'SeriesDescription'):
            metadata['series_description'] = str(dcm.SeriesDescription)
        if hasattr(dcm, 'StudyDate'):
            metadata['study_date'] = str(dcm.StudyDate)
        
        return metadata
    
    def create_affine_matrix(self, metadata: dict, volume_shape: Tuple) -> np.ndarray:
        """
        创建仿射变换矩阵
        
        参数:
            metadata: 元数据字典
            volume_shape: 体数据形状
            
        返回:
            4x4仿射变换矩阵
        """
        pixel_spacing = metadata.get('pixel_spacing', [1.0, 1.0])
        slice_thickness = metadata.get('slice_thickness', 1.0)
        
        # 创建仿射矩阵 (RAS+ 坐标系统)
        affine = np.eye(4)
        affine[0, 0] = pixel_spacing[1]  # 列间距 (x)
        affine[1, 1] = pixel_spacing[0]  # 行间距 (y)
        affine[2, 2] = slice_thickness    # 切片间距 (z)
        
        return affine
    
    def select_roi_interactive(self, volume: np.ndarray,
                              roi_size: Tuple[int, int, int] = (512, 512, 256)) -> Dict[str, Tuple[int, int]]:
        """
        Interactive ROI selection with fixed size (512×512×512)

        Parameters:
            volume: 3D numpy array (z, y, x)
            roi_size: Fixed ROI size (x_size, y_size, z_size)
            
        Returns:
            ROI boundary dictionary {'x': (x_min, x_max), 'y': (y_min, y_max), 'z': (z_min, z_max)}
        """
        print("\n=== Interactive ROI Selection (Fixed Size 512×512×512) ===")
        print("Instructions:")
        print("  - Click on any view to move the 512×512×512 fixed selection box")
        print("  - Red box shows current selection area")
        print("  - Close window to complete selection\n")
        
        z_dim, y_dim, x_dim = volume.shape
        roi_x_size, roi_y_size, roi_z_size = roi_size
        
        # Check if volume size is sufficient
        if x_dim < roi_x_size or y_dim < roi_y_size or z_dim < roi_z_size:
            print(f"Warning: Volume size ({x_dim}×{y_dim}×{z_dim}) is smaller than ROI size ({roi_x_size}×{roi_y_size}×{roi_z_size})")
            print(f"Using actual volume size instead")
            roi_x_size = min(roi_x_size, x_dim)
            roi_y_size = min(roi_y_size, y_dim)
            roi_z_size = min(roi_z_size, z_dim)
        
        # 初始化ROI中心位置（体块中心）
        roi_center = {
            'x': x_dim // 2,
            'y': y_dim // 2,
            'z': z_dim // 2
        }
        
        # 计算初始ROI边界
        def update_roi_bounds():
            x_min = max(0, roi_center['x'] - roi_x_size // 2)
            x_max = min(x_dim, x_min + roi_x_size)
            x_min = x_max - roi_x_size  # 调整确保大小正确
            
            y_min = max(0, roi_center['y'] - roi_y_size // 2)
            y_max = min(y_dim, y_min + roi_y_size)
            y_min = y_max - roi_y_size
            
            z_min = max(0, roi_center['z'] - roi_z_size // 2)
            z_max = min(z_dim, z_min + roi_z_size)
            z_min = z_max - roi_z_size
            
            return {
                'x': [x_min, x_max],
                'y': [y_min, y_max],
                'z': [z_min, z_max]
            }
        
        roi = update_roi_bounds()
        
        # Create three-view display
        fig = plt.figure(figsize=(15, 5))
        fig.suptitle('Click to Select ROI Area (Close Window to Complete)', fontsize=14, fontweight='bold')
        
        # 计算中间切片
        mid_z = z_dim // 2
        mid_y = y_dim // 2
        mid_x = x_dim // 2
        
        # Axial view (xy plane)
        ax1 = fig.add_subplot(131)
        axial_slice = volume[mid_z, :, :]
        im1 = ax1.imshow(axial_slice, cmap='gray', aspect='auto')
        ax1.set_title(f'Axial View (Z={mid_z})\nSelect X-Y Range', fontsize=10)
        ax1.set_xlabel('X Axis')
        ax1.set_ylabel('Y Axis')
        plt.colorbar(im1, ax=ax1, fraction=0.046)
        
        # Coronal view (xz plane)
        ax2 = fig.add_subplot(132)
        coronal_slice = volume[:, mid_y, :]
        im2 = ax2.imshow(coronal_slice, cmap='gray', aspect='auto')
        ax2.set_title(f'Coronal View (Y={mid_y})\nSelect X-Z Range', fontsize=10)
        ax2.set_xlabel('X Axis')
        ax2.set_ylabel('Z Axis')
        plt.colorbar(im2, ax=ax2, fraction=0.046)
        
        # Sagittal view (yz plane)
        ax3 = fig.add_subplot(133)
        sagittal_slice = volume[:, :, mid_x]
        im3 = ax3.imshow(sagittal_slice, cmap='gray', aspect='auto')
        ax3.set_title(f'Sagittal View (X={mid_x})\nSelect Y-Z Range', fontsize=10)
        ax3.set_xlabel('Y Axis')
        ax3.set_ylabel('Z Axis')
        plt.colorbar(im3, ax=ax3, fraction=0.046)
        
        # 绘制固定ROI矩形框
        rect1 = Rectangle((roi['x'][0], roi['y'][0]), roi_x_size, roi_y_size,
                         linewidth=2, edgecolor='red', facecolor='none')
        rect2 = Rectangle((roi['x'][0], roi['z'][0]), roi_x_size, roi_z_size,
                         linewidth=2, edgecolor='red', facecolor='none')
        rect3 = Rectangle((roi['y'][0], roi['z'][0]), roi_y_size, roi_z_size,
                         linewidth=2, edgecolor='red', facecolor='none')
        
        ax1.add_patch(rect1)
        ax2.add_patch(rect2)
        ax3.add_patch(rect3)
        
        # Function to update all rectangle boxes
        def update_all_rects():
            """Update rectangle positions in all three views, ensuring coordinate consistency"""
            rect1.set_xy((roi['x'][0], roi['y'][0]))
            rect2.set_xy((roi['x'][0], roi['z'][0]))
            rect3.set_xy((roi['y'][0], roi['z'][0]))
            fig.canvas.draw_idle()
        
        # Click callback functions to move ROI
        def onclick_axial(event):
            if event.xdata is None or event.ydata is None:
                return
            
            # Update center position (only X and Y, keep Z unchanged)
            roi_center['x'] = int(event.xdata)
            roi_center['y'] = int(event.ydata)
            
            # Update ROI boundaries
            nonlocal roi
            roi = update_roi_bounds()
            
            # Update all three view rectangles
            update_all_rects()
            
            print(f"Axial view clicked: Center moved to X={roi_center['x']}, Y={roi_center['y']}, Z={roi_center['z']}")
            print(f"  Full ROI: X=[{roi['x'][0]}, {roi['x'][1]}], Y=[{roi['y'][0]}, {roi['y'][1]}], Z=[{roi['z'][0]}, {roi['z'][1]}]")
        
        def onclick_coronal(event):
            if event.xdata is None or event.ydata is None:
                return
            
            # Update center position (only X and Z, keep Y unchanged)
            roi_center['x'] = int(event.xdata)
            roi_center['z'] = int(event.ydata)
            
            nonlocal roi
            roi = update_roi_bounds()
            
            # Update all three view rectangles
            update_all_rects()
            
            print(f"Coronal view clicked: Center moved to X={roi_center['x']}, Y={roi_center['y']}, Z={roi_center['z']}")
            print(f"  Full ROI: X=[{roi['x'][0]}, {roi['x'][1]}], Y=[{roi['y'][0]}, {roi['y'][1]}], Z=[{roi['z'][0]}, {roi['z'][1]}]")
        
        def onclick_sagittal(event):
            if event.xdata is None or event.ydata is None:
                return
            
            # Update center position (only Y and Z, keep X unchanged)
            roi_center['y'] = int(event.xdata)
            roi_center['z'] = int(event.ydata)
            
            nonlocal roi
            roi = update_roi_bounds()
            
            # Update all three view rectangles
            update_all_rects()
            
            print(f"Sagittal view clicked: Center moved to X={roi_center['x']}, Y={roi_center['y']}, Z={roi_center['z']}")
            print(f"  Full ROI: X=[{roi['x'][0]}, {roi['x'][1]}], Y=[{roi['y'][0]}, {roi['y'][1]}], Z=[{roi['z'][0]}, {roi['z'][1]}]")
        
        # Connect click events
        fig.canvas.mpl_connect('button_press_event',
                              lambda event: onclick_axial(event) if event.inaxes == ax1 else None)
        fig.canvas.mpl_connect('button_press_event',
                              lambda event: onclick_coronal(event) if event.inaxes == ax2 else None)
        fig.canvas.mpl_connect('button_press_event',
                              lambda event: onclick_sagittal(event) if event.inaxes == ax3 else None)
        
        plt.tight_layout()
        plt.show()
        
        # Convert to tuple format
        roi_bounds = {
            'x': (roi['x'][0], roi['x'][1]),
            'y': (roi['y'][0], roi['y'][1]),
            'z': (roi['z'][0], roi['z'][1])
        }
        
        print(f"\nFinal ROI Selection (Fixed Size {roi_x_size}×{roi_y_size}×{roi_z_size}):")
        print(f"  X Range: {roi_bounds['x'][0]} - {roi_bounds['x'][1]}")
        print(f"  Y Range: {roi_bounds['y'][0]} - {roi_bounds['y'][1]}")
        print(f"  Z Range: {roi_bounds['z'][0]} - {roi_bounds['z'][1]}")
        print(f"  Center Position: X={roi_center['x']}, Y={roi_center['y']}, Z={roi_center['z']}")
        
        return roi_bounds
    
    def crop_volume(self, volume: np.ndarray, roi: Dict[str, Tuple[int, int]]) -> np.ndarray:
        """
        根据ROI裁剪体块，并将ROI坐标保存到YAML文件
        
        参数:
            volume: 原始3D数组 (z, y, x)
            roi: ROI边界字典
            
        返回:
            裁剪后的3D数组
        """
        z_min, z_max = roi['z']
        y_min, y_max = roi['y']
        x_min, x_max = roi['x']
        
        cropped = volume[z_min:z_max, y_min:y_max, x_min:x_max]
        
        print(f"\n裁剪完成:")
        print(f"  原始形状: {volume.shape}")
        print(f"  裁剪后形状: {cropped.shape}")
        print(f"  数据范围: [{cropped.min():.2f}, {cropped.max():.2f}] HU")
        
        # 保存ROI坐标到YAML文件
        roi_info = {
            'roi_coordinates': {
                'x': {'min': int(x_min), 'max': int(x_max), 'size': int(x_max - x_min)},
                'y': {'min': int(y_min), 'max': int(y_max), 'size': int(y_max - y_min)},
                'z': {'min': int(z_min), 'max': int(z_max), 'size': int(z_max - z_min)}
            },
            'original_volume_shape': {
                'z': int(volume.shape[0]),
                'y': int(volume.shape[1]),
                'x': int(volume.shape[2])
            },
            'cropped_volume_shape': {
                'z': int(cropped.shape[0]),
                'y': int(cropped.shape[1]),
                'x': int(cropped.shape[2])
            }
        }
        
        yaml_path = os.path.join(self.output_folder, 'hu_adjustment_stats.yaml')
        
        # 如果文件已存在，追加ROI信息；否则创建新文件
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r', encoding='utf-8') as f:
                existing_data = yaml.safe_load(f) or {}
            existing_data.update(roi_info)
            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(existing_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        else:
            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(roi_info, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        
        print(f"  ✓ ROI坐标已保存到: {yaml_path}")
        
        return cropped
    
    def adjust_hu_ranges(self, volume: np.ndarray,
                        high_threshold: float = 1000.0,
                        high_boost: float = 5000.0,
                        low_reduce: float = 500.0,
                        save_stats: bool = True) -> np.ndarray:
        """
        调整HU值范围：对像素值>1000的加5000，对像素值≤1000的减500
        
        参数:
            volume: 输入3D numpy数组 (HU值)
            high_threshold: 高低值分界阈值，默认 1000
            high_boost: 高值增加量，默认 5000
            low_reduce: 低值减少量，默认 500
            save_stats: 是否保存统计信息到YAML文件
            
        返回:
            调整后的3D数组
        """
        print(f"\n调整HU值范围...")
        print(f"  原始数据范围: [{volume.min():.2f}, {volume.max():.2f}] HU")
        
        adjusted = volume.copy()
        
        # 处理高值（>1000）：加5000
        high_mask = volume > high_threshold
        high_count = np.sum(high_mask)
        
        if high_count > 0:
            print(f"  高值区域 (>{high_threshold:.0f} HU):")
            print(f"    - 体素数: {high_count} ({high_count/volume.size*100:.2f}%)")
            print(f"    - 增加值: +{high_boost:.0f}")
            adjusted[high_mask] = volume[high_mask] + high_boost
            actual_increase = adjusted[high_mask].mean() - volume[high_mask].mean()
            print(f"    - 实际平均增加: +{actual_increase:.2f}")
        
        # 处理低值（≤1000）：减500
        low_mask = volume <= high_threshold
        low_count = np.sum(low_mask)
        
        if low_count > 0:
            print(f"  低值区域 (≤{high_threshold:.0f} HU):")
            print(f"    - 体素数: {low_count} ({low_count/volume.size*100:.2f}%)")
            print(f"    - 减少值: -{low_reduce:.0f}")
            adjusted[low_mask] = volume[low_mask] - low_reduce
            actual_decrease = volume[low_mask].mean() - adjusted[low_mask].mean()
            print(f"    - 实际平均减少: -{actual_decrease:.2f}")
        
        # 获取调整后的最大值和最小值
        adjusted_min = float(adjusted.min())
        adjusted_max = float(adjusted.max())
        
        print(f"  调整后范围: [{adjusted_min:.2f}, {adjusted_max:.2f}] HU")
        print(f"  ✓ HU值范围调整完成")
        
        # 保存统计信息到YAML文件
        if save_stats:
            stats = {
                'original_range': {
                    'min': float(volume.min()),
                    'max': float(volume.max()),
                    'mean': float(volume.mean()),
                    'std': float(volume.std())
                },
                'adjusted_range': {
                    'min': adjusted_min,
                    'max': adjusted_max,
                    'mean': float(adjusted.mean()),
                    'std': float(adjusted.std())
                },
                'adjustment_params': {
                    'high_threshold': high_threshold,
                    'high_boost': high_boost,
                    'low_reduce': low_reduce
                },
                'voxel_counts': {
                    'high_voxels': int(high_count),
                    'low_voxels': int(low_count),
                    'total_voxels': int(volume.size)
                }
            }
            
            yaml_path = os.path.join(self.output_folder, 'hu_adjustment_stats.yaml')
            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(stats, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            
            print(f"  ✓ 统计信息已保存到: {yaml_path}")
        
        return adjusted
    
    def enhance_contrast(self, volume: np.ndarray,
                        method: str = 'window',
                        tissue_boost: float = 200.0,
                        air_threshold: float = -500.0) -> np.ndarray:
        """
        增强CT图像对比度，使组织与空气的区别更明显
        
        参数:
            volume: 输入3D numpy数组 (HU值)
            method: 增强方法
                - 'window': 窗位窗宽调整（推荐）
                - 'boost': 组织区域增强
                - 'sigmoid': S型曲线增强
                - 'clahe': 对比度受限自适应直方图均衡化
                - 'log': 对数变换增强
            tissue_boost: 组织增强值（仅用于'boost'方法）
            air_threshold: 空气阈值，低于此值认为是空气（HU值）
            
        返回:
            增强后的3D数组
        """
        print(f"\n对比度增强...")
        print(f"  方法: {method}")
        print(f"  原始数据范围: [{volume.min():.2f}, {volume.max():.2f}] HU")
        
        enhanced = volume.copy()
        
        if method == 'window':
            # 窗位窗宽调整 - 针对软组织优化
            # 窗位(window level): 选择感兴趣的HU值中心
            # 窗宽(window width): 控制显示的HU值范围
            window_center = 40  # 软组织窗位
            window_width = 400  # 软组织窗宽
            
            window_min = window_center - window_width / 2
            window_max = window_center + window_width / 2
            
            print(f"  窗位: {window_center} HU")
            print(f"  窗宽: {window_width} HU")
            print(f"  显示范围: [{window_min:.1f}, {window_max:.1f}] HU")
            
            # 将窗外的值裁剪到窗范围内
            enhanced = np.clip(volume, window_min, window_max)
            
            # 线性拉伸到全范围以增强对比度
            enhanced = (enhanced - window_min) / (window_max - window_min) * 2000 - 1000
            
        elif method == 'boost':
            # 组织区域增强 - 给组织加权值
            print(f"  空气阈值: {air_threshold} HU")
            print(f"  组织增强值: +{tissue_boost} HU")
            
            # 识别组织区域（大于空气阈值）
            tissue_mask = volume > air_threshold
            
            # 对组织区域增强
            enhanced[tissue_mask] = volume[tissue_mask] + tissue_boost
            
            tissue_ratio = np.sum(tissue_mask) / tissue_mask.size
            print(f"  组织占比: {tissue_ratio:.1%}")
            
        elif method == 'sigmoid':
            # S型曲线增强 - 增强中间值，抑制极端值
            # 使用Sigmoid函数平滑地增强对比度
            center = 0  # 增强的中心点（HU值）
            slope = 0.01  # 曲线斜率，越大对比度越强
            
            print(f"  中心点: {center} HU")
            print(f"  斜率参数: {slope}")
            
            # Sigmoid变换
            normalized = (volume - center) * slope
            enhanced = 2000 / (1 + np.exp(-normalized)) - 1000
            
        elif method == 'clahe':
            # 对比度受限自适应直方图均衡化
            # 注意：这个方法需要先归一化到0-255范围
            print(f"  使用CLAHE增强")
            
            # 先归一化到0-1范围
            vol_min, vol_max = volume.min(), volume.max()
            normalized = (volume - vol_min) / (vol_max - vol_min + 1e-10)
            
            # 逐切片应用CLAHE
            enhanced_slices = []
            from skimage import exposure
            
            for i in range(volume.shape[0]):
                slice_data = normalized[i, :, :]
                # CLAHE参数：clip_limit控制对比度限制
                enhanced_slice = exposure.equalize_adapthist(
                    slice_data,
                    clip_limit=0.03,
                    kernel_size=None
                )
                enhanced_slices.append(enhanced_slice)
            
            enhanced = np.stack(enhanced_slices, axis=0)
            # 恢复到HU值范围
            enhanced = enhanced * (vol_max - vol_min) + vol_min
            
        elif method == 'log':
            # 对数变换 - 压缩高值，增强低值
            print(f"  使用对数变换增强对比度")
            
            # 将数据移到正值范围（对数要求正数）
            data_min = volume.min()
            shifted = volume - data_min + 1.0  # 加1避免log(0)
            
            print(f"  原始范围: [{volume.min():.2f}, {volume.max():.2f}]")
            print(f"  平移后范围: [{shifted.min():.2f}, {shifted.max():.2f}]")
            
            # 应用对数变换
            log_transformed = np.log(shifted + 1.0)
            
            print(f"  对数变换后范围: [{log_transformed.min():.4f}, {log_transformed.max():.4f}]")
            
            # 归一化到(-1000, 1000)范围
            log_min = log_transformed.min()
            log_max = log_transformed.max()
            
            if log_max - log_min > 1e-10:
                enhanced = (log_transformed - log_min) / (log_max - log_min) * 2000 - 1000
            else:
                enhanced = np.zeros_like(volume)
            
        else:
            raise ValueError(f"不支持的增强方法: {method}. "
                           f"支持的方法: 'window', 'boost', 'sigmoid', 'clahe', 'log'")
        
        print(f"  增强后范围: [{enhanced.min():.2f}, {enhanced.max():.2f}] HU")
        print(f"  ✓ 对比度增强完成")
        
        return enhanced
    
    def normalize_and_convert(self, volume: np.ndarray,
                             target_range: Tuple[float, float] = (-1000.0, 1000.0),
                             output_dtype: np.dtype = np.int16) -> np.ndarray:
        """
        归一化数据并转换数据类型
        
        参数:
            volume: 输入3D numpy数组
            target_range: 目标数值范围 (min, max)
            output_dtype: 输出数据类型
            
        返回:
            归一化并转换后的3D数组
        """
        print(f"\n数据归一化和类型转换...")
        print(f"  原始数据范围: [{volume.min():.2f}, {volume.max():.2f}]")
        print(f"  原始数据类型: {volume.dtype}")
        
        # 获取当前数据范围
        data_min = volume.min()
        data_max = volume.max()
        
        # 归一化到目标范围
        target_min, target_max = target_range
        
        if data_max - data_min > 1e-6:  # 避免除零
            # 线性映射: [data_min, data_max] -> [target_min, target_max]
            normalized = (volume - data_min) / (data_max - data_min) * (target_max - target_min) + target_min
        else:
            # 如果数据范围太小，直接设置为目标范围的中间值
            normalized = np.full_like(volume, (target_min + target_max) / 2)
        
        # 转换为目标数据类型
        # 对于整数类型，需要先裁剪到合理范围再转换
        if np.issubdtype(output_dtype, np.integer):
            # 获取目标类型的范围
            dtype_info = np.iinfo(output_dtype)
            # 裁剪到目标类型能表示的范围
            clipped_min = max(target_min, dtype_info.min)
            clipped_max = min(target_max, dtype_info.max)
            normalized = np.clip(normalized, clipped_min, clipped_max)
        
        # 转换数据类型
        converted = normalized.astype(output_dtype)
        
        print(f"  归一化后范围: [{converted.min():.2f}, {converted.max():.2f}]")
        print(f"  目标数据类型: {converted.dtype}")
        print(f"  内存占用: {converted.nbytes / (1024*1024):.2f} MB")
        
        return converted
    
    def save_as_nifti(self, volume: np.ndarray, metadata: dict,
                     output_filename: str = "output.nii.gz",
                     normalize: bool = True,
                     target_range: Tuple[float, float] = (-1000.0, 1000.0),
                     output_dtype: np.dtype = np.int16) -> str:
        """
        保存为NIFTI格式
        
        参数:
            volume: 3D numpy数组
            metadata: 元数据字典
            output_filename: 输出文件名
            normalize: 是否进行归一化和类型转换
            target_range: 归一化目标范围
            output_dtype: 输出数据类型
            
        返回:
            输出文件路径
        """
        print(f"\n保存为NIFTI格式...")
        
        # 归一化和类型转换
        if normalize:
            volume = self.normalize_and_convert(volume, target_range, output_dtype)
        
        # 调整维度顺序: (z, y, x) -> (x, y, z)
        volume_transposed = np.transpose(volume, (2, 1, 0))
        
        # 创建仿射矩阵
        affine = self.create_affine_matrix(metadata, volume.shape)
        
        # 创建NIFTI图像对象
        nifti_img = nib.Nifti1Image(volume_transposed, affine)
        
        # 设置头部信息
        header = nifti_img.header
        header.set_xyzt_units('mm', 'sec')
        
        # 添加描述信息
        if 'series_description' in metadata:
            header['descrip'] = metadata['series_description'].encode('utf-8')[:80]
        
        # 保存文件
        output_path = os.path.join(self.output_folder, output_filename)
        nib.save(nifti_img, output_path)
        
        print(f"✓ 已保存: {output_path}")
        print(f"  形状: {volume_transposed.shape} (x×y×z)")
        print(f"  体素大小: {metadata['pixel_spacing'][1]:.3f} × {metadata['pixel_spacing'][0]:.3f} × {metadata['slice_thickness']:.3f} mm³")
        print(f"  数据类型: {volume_transposed.dtype}")
        print(f"  数据范围: [{volume.min():.2f}, {volume.max():.2f}] HU")
        
        return output_path
    
    def convert(self, output_filename: str = "output.nii.gz",
                enable_roi_selection: bool = False,
                enable_contrast_enhancement: bool = False,
                contrast_method: str = 'window',
                enable_hu_adjustment: bool = False,
                high_boost: float = 2000.0,
                low_reduce: float = 500.0) -> str:
        """
        执行转换
        
        参数:
            output_filename: 输出文件名
            enable_roi_selection: 是否启用ROI选择
            enable_contrast_enhancement: 是否启用对比度增强
            contrast_method: 对比度增强方法 ('window', 'boost', 'sigmoid', 'clahe')
            enable_hu_adjustment: 是否启用HU值范围调整
            high_boost: 高值范围(1000-5000)增加的值，默认2000
            low_reduce: 低值范围(300-1000)减少的值，默认500
            
        返回:
            输出文件路径
        """
        try:
            # 1. 收集DICOM文件
            dicom_files = self.collect_dicom_files()
            
            # 2. 读取DICOM序列
            volume, metadata = self.read_dicom_series(dicom_files)
            
            # 3. Y轴翻转
            print(f"\n执行Y轴翻转...")
            print(f"  翻转前形状: {volume.shape}")
            volume = np.flip(volume, axis=1)  # axis=1 对应 y 轴
            print(f"  翻转后形状: {volume.shape}")
            print(f"  ✓ Y轴翻转完成")
            
            # 4. HU值范围调整(如果启用)
            if enable_hu_adjustment:
                volume = self.adjust_hu_ranges(
                    volume,
                    high_boost=high_boost,
                    low_reduce=low_reduce,
                    save_stats=True
                )
            
            # 5. 对比度增强(如果启用)
            if enable_contrast_enhancement:
                volume = self.enhance_contrast(volume, method=contrast_method)
            
            # 6. ROI选择(如果启用)
            if enable_roi_selection:
                roi = self.select_roi_interactive(volume)
                volume = self.crop_volume(volume, roi)
            
            # 7. 保存为NIFTI (归一化到(-1000, 1000)并转换为int16)
            output_path = self.save_as_nifti(
                volume, metadata, output_filename,
                normalize=True,
                target_range=(-1000.0, 1000.0),
                output_dtype=np.int16
            )
            
            print("\n转换完成!")
            return output_path
            
        except Exception as e:
            print(f"\n错误: {e}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='DICOM to NIFTI 转换工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本使用
  python dcmcompose.py -i /path/to/dicom/folder
  
  # 指定输出文件名
  python dcmcompose.py -i /path/to/dicom/folder -o my_scan.nii.gz
  
  # 指定输出目录
  python dcmcompose.py -i /path/to/dicom/folder -d ./output
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='DICOM文件所在文件夹路径'
    )
    parser.add_argument(
        '-o', '--output',
        default='output.nii.gz',
        help='输出NIFTI文件名 (默认: output.nii.gz)'
    )
    parser.add_argument(
        '-d', '--output-dir',
        default='./nifti_output',
        help='输出文件夹路径 (默认: ./nifti_output)'
    )
    parser.add_argument(
        '--roi',
        action='store_true',
        help='启用交互式ROI选择'
    )
    parser.add_argument(
        '--enhance',
        action='store_true',
        help='启用对比度增强'
    )
    parser.add_argument(
        '--contrast-method',
        choices=['window', 'boost', 'sigmoid', 'clahe', 'log'],
        default='window',
        help='对比度增强方法 (默认: window)'
    )
    parser.add_argument(
        '--adjust-hu',
        action='store_true',
        help='启用HU值范围调整'
    )
    parser.add_argument(
        '--high-boost',
        type=float,
        default=5000.0,
        help='高值区域(>1000)增加的值 (默认: 5000)'
    )
    parser.add_argument(
        '--low-reduce',
        type=float,
        default=500.0,
        help='低值区域(≤1000)减少的值 (默认: 500)'
    )
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.exists(args.input):
        print(f"错误: 输入目录不存在: {args.input}")
        return
    
    # 创建转换器并执行转换
    converter = DicomToNiftiConverter(args.input, args.output_dir)
    converter.convert(
        args.output,
        enable_roi_selection=args.roi,
        enable_contrast_enhancement=args.enhance,
        contrast_method=args.contrast_method,
        enable_hu_adjustment=args.adjust_hu,
        high_boost=args.high_boost,
        low_reduce=args.low_reduce
    )


if __name__ == "__main__":
    main()