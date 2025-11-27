#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NIFTI文件查看器
读取并窗口化展示NIFTI文件的三视图
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import argparse
import os


class NiftiViewer:
    """NIFTI文件查看器"""
    
    def __init__(self, nifti_file: str):
        """
        初始化查看器
        
        参数:
            nifti_file: NIFTI文件路径
        """
        if not os.path.exists(nifti_file):
            raise FileNotFoundError(f"文件不存在: {nifti_file}")
        
        self.nifti_file = nifti_file
        self.img = None
        self.data = None
        self.affine = None
        self.header = None
        
    def load_nifti(self):
        """加载NIFTI文件"""
        print(f"加载文件: {self.nifti_file}")
        
        # 读取NIFTI文件
        self.img = nib.load(self.nifti_file)
        self.data = self.img.get_fdata()
        self.affine = self.img.affine
        self.header = self.img.header
        
        # 显示基本信息
        print(f"\nFile Information:")
        print(f"  Shape: {self.data.shape}")
        print(f"  Data Type: {self.data.dtype}")
        print(f"  Data Range: [{self.data.min():.2f}, {self.data.max():.2f}]")
        
        # 体素尺寸
        voxel_sizes = self.header.get_zooms()[:3]
        print(f"  Voxel Size: {voxel_sizes[0]:.3f} x {voxel_sizes[1]:.3f} x {voxel_sizes[2]:.3f} mm^3")
        
        # 如果是4D数据，只取第一个时间点
        if len(self.data.shape) == 4:
            print(f"  Detected 4D data, using first timepoint (total {self.data.shape[3]} timepoints)")
            self.data = self.data[:, :, :, 0]
        
        return self.data
    
    def show_interactive(self):
        """交互式显示三视图（带ROI框选功能）"""
        if self.data is None:
            self.load_nifti()
        
        # 数据维度 (x, y, z)
        x_dim, y_dim, z_dim = self.data.shape
        
        # 初始切片位置
        x_slice = x_dim // 2
        y_slice = y_dim // 2
        z_slice = z_dim // 2
        
        # ROI状态
        self.roi_data = {
            'sagittal': None,  # (x1, y1, x2, y2)
            'coronal': None,
            'axial': None
        }
        
        # 创建图形和子图
        fig = plt.figure(figsize=(16, 7))
        fig.suptitle(f'NIFTI Viewer with ROI Selection - {os.path.basename(self.nifti_file)}',
                     fontsize=14, fontweight='bold')
        
        # 矢状面 (Sagittal) - YZ平面
        ax1 = plt.subplot(231)
        sagittal_slice = self.data[x_slice, :, :]
        im1 = ax1.imshow(np.rot90(sagittal_slice), cmap='gray', aspect='auto')
        ax1.set_title(f'Sagittal View\nX = {x_slice}', fontsize=11)
        ax1.set_xlabel('Z axis')
        ax1.set_ylabel('Y axis')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # 冠状面 (Coronal) - XZ平面
        ax2 = plt.subplot(232)
        coronal_slice = self.data[:, y_slice, :]
        im2 = ax2.imshow(np.rot90(coronal_slice), cmap='gray', aspect='auto')
        ax2.set_title(f'Coronal View\nY = {y_slice}', fontsize=11)
        ax2.set_xlabel('Z axis')
        ax2.set_ylabel('X axis')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        # 轴向面 (Axial) - XY平面
        ax3 = plt.subplot(233)
        axial_slice = self.data[:, :, z_slice]
        im3 = ax3.imshow(np.rot90(axial_slice), cmap='gray', aspect='auto')
        ax3.set_title(f'Axial View\nZ = {z_slice}', fontsize=11)
        ax3.set_xlabel('X axis')
        ax3.set_ylabel('Y axis')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        
        # ROI统计信息显示区域
        ax_stats1 = plt.subplot(234)
        ax_stats1.axis('off')
        stats_text1 = ax_stats1.text(0.05, 0.5, 'Sagittal ROI:\nNot selected',
                                     fontsize=10, verticalalignment='center',
                                     family='monospace')
        
        ax_stats2 = plt.subplot(235)
        ax_stats2.axis('off')
        stats_text2 = ax_stats2.text(0.05, 0.5, 'Coronal ROI:\nNot selected',
                                     fontsize=10, verticalalignment='center',
                                     family='monospace')
        
        ax_stats3 = plt.subplot(236)
        ax_stats3.axis('off')
        stats_text3 = ax_stats3.text(0.05, 0.5, 'Axial ROI:\nNot selected',
                                     fontsize=10, verticalalignment='center',
                                     family='monospace')
        
        # 调整子图布局
        plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.18,
                          wspace=0.3, hspace=0.3)
        
        # ROI矩形对象
        self.roi_rects = {
            'sagittal': None,
            'coronal': None,
            'axial': None
        }
        
        # 创建滑块控制切片位置
        # 矢状面滑块
        ax_slider1 = plt.axes([0.15, 0.12, 0.2, 0.03])
        slider1 = Slider(ax_slider1, 'X (Sagittal)', 0, x_dim-1,
                        valinit=x_slice, valstep=1, color='#ff7f0e')
        
        # 冠状面滑块
        ax_slider2 = plt.axes([0.42, 0.12, 0.2, 0.03])
        slider2 = Slider(ax_slider2, 'Y (Coronal)', 0, y_dim-1,
                        valinit=y_slice, valstep=1, color='#2ca02c')
        
        # 轴向面滑块
        ax_slider3 = plt.axes([0.69, 0.12, 0.2, 0.03])
        slider3 = Slider(ax_slider3, 'Z (Axial)', 0, z_dim-1,
                        valinit=z_slice, valstep=1, color='#d62728')
        
        # 滑块更新函数
        def update_sagittal(val):
            x = int(slider1.val)
            sagittal_slice = self.data[x, :, :]
            im1.set_data(np.rot90(sagittal_slice))
            ax1.set_title(f'Sagittal View\nX = {x}', fontsize=11)
            fig.canvas.draw_idle()
        
        def update_coronal(val):
            y = int(slider2.val)
            coronal_slice = self.data[:, y, :]
            im2.set_data(np.rot90(coronal_slice))
            ax2.set_title(f'Coronal View\nY = {y}', fontsize=11)
            fig.canvas.draw_idle()
        
        def update_axial(val):
            z = int(slider3.val)
            axial_slice = self.data[:, :, z]
            im3.set_data(np.rot90(axial_slice))
            ax3.set_title(f'Axial View\nZ = {z}', fontsize=11)
            fig.canvas.draw_idle()
        
        # ROI选择回调函数
        def calculate_roi_stats(slice_data, x1, y1, x2, y2):
            """计算ROI区域的统计信息"""
            # 确保坐标在有效范围内
            h, w = slice_data.shape
            x1, x2 = max(0, min(x1, x2)), min(w, max(x1, x2))
            y1, y2 = max(0, min(y1, y2)), min(h, max(y1, y2))
            
            if x2 <= x1 or y2 <= y1:
                return None
            
            # 由于图像被旋转90度，需要调整坐标
            roi_data = slice_data[y1:y2, x1:x2]
            
            if roi_data.size == 0:
                return None
            
            stats = {
                'mean': np.mean(roi_data),
                'std': np.std(roi_data),
                'min': np.min(roi_data),
                'max': np.max(roi_data),
                'median': np.median(roi_data),
                'size': roi_data.size,
                'shape': roi_data.shape
            }
            return stats
        
        def format_stats_text(view_name, stats, coords):
            """Format statistics text"""
            if stats is None:
                return f'{view_name} ROI:\nNot selected or invalid'
            
            text = f'{view_name} ROI:\n'
            text += f'Coords: ({coords[0]:.0f},{coords[1]:.0f}) - ({coords[2]:.0f},{coords[3]:.0f})\n'
            text += f'Size: {stats["shape"][0]}×{stats["shape"][1]} ({stats["size"]} voxels)\n'
            text += f'━━━━━━━━━━━━━━━━━━━━\n'
            text += f'Mean:   {stats["mean"]:>8.2f}\n'
            text += f'StdDev: {stats["std"]:>8.2f}\n'
            text += f'Median: {stats["median"]:>8.2f}\n'
            text += f'Min:    {stats["min"]:>8.2f}\n'
            text += f'Max:    {stats["max"]:>8.2f}\n'
            return text
        
        def onselect_sagittal(eclick, erelease):
            """矢状面ROI选择"""
            x1, y1 = int(eclick.xdata), int(eclick.ydata)
            x2, y2 = int(erelease.xdata), int(erelease.ydata)
            
            # 保存ROI坐标
            self.roi_data['sagittal'] = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            
            # 获取当前切片数据
            current_x = int(slider1.val)
            sagittal_slice = self.data[current_x, :, :]
            rotated = np.rot90(sagittal_slice)
            
            # 计算统计信息
            stats = calculate_roi_stats(rotated, min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            
            # 更新显示
            stats_text1.set_text(format_stats_text('Sagittal', stats, self.roi_data['sagittal']))
            
            # 绘制ROI矩形
            if self.roi_rects['sagittal'] is not None:
                self.roi_rects['sagittal'].remove()
            
            from matplotlib.patches import Rectangle
            rect = Rectangle((min(x1, x2), min(y1, y2)),
                           abs(x2-x1), abs(y2-y1),
                           linewidth=2, edgecolor='red', facecolor='none')
            self.roi_rects['sagittal'] = ax1.add_patch(rect)
            fig.canvas.draw_idle()
        
        def onselect_coronal(eclick, erelease):
            """冠状面ROI选择"""
            x1, y1 = int(eclick.xdata), int(eclick.ydata)
            x2, y2 = int(erelease.xdata), int(erelease.ydata)
            
            self.roi_data['coronal'] = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            
            current_y = int(slider2.val)
            coronal_slice = self.data[:, current_y, :]
            rotated = np.rot90(coronal_slice)
            
            stats = calculate_roi_stats(rotated, min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            stats_text2.set_text(format_stats_text('Coronal', stats, self.roi_data['coronal']))
            
            if self.roi_rects['coronal'] is not None:
                self.roi_rects['coronal'].remove()
            
            from matplotlib.patches import Rectangle
            rect = Rectangle((min(x1, x2), min(y1, y2)),
                           abs(x2-x1), abs(y2-y1),
                           linewidth=2, edgecolor='red', facecolor='none')
            self.roi_rects['coronal'] = ax2.add_patch(rect)
            fig.canvas.draw_idle()
        
        def onselect_axial(eclick, erelease):
            """轴向面ROI选择"""
            x1, y1 = int(eclick.xdata), int(eclick.ydata)
            x2, y2 = int(erelease.xdata), int(erelease.ydata)
            
            self.roi_data['axial'] = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            
            current_z = int(slider3.val)
            axial_slice = self.data[:, :, current_z]
            rotated = np.rot90(axial_slice)
            
            stats = calculate_roi_stats(rotated, min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            stats_text3.set_text(format_stats_text('Axial', stats, self.roi_data['axial']))
            
            if self.roi_rects['axial'] is not None:
                self.roi_rects['axial'].remove()
            
            from matplotlib.patches import Rectangle
            rect = Rectangle((min(x1, x2), min(y1, y2)),
                           abs(x2-x1), abs(y2-y1),
                           linewidth=2, edgecolor='red', facecolor='none')
            self.roi_rects['axial'] = ax3.add_patch(rect)
            fig.canvas.draw_idle()
        
        # 创建矩形选择器
        from matplotlib.widgets import RectangleSelector
        
        rs1 = RectangleSelector(ax1, onselect_sagittal,
                               useblit=True, button=[1],
                               minspanx=5, minspany=5,
                               spancoords='pixels',
                               interactive=True)
        
        rs2 = RectangleSelector(ax2, onselect_coronal,
                               useblit=True, button=[1],
                               minspanx=5, minspany=5,
                               spancoords='pixels',
                               interactive=True)
        
        rs3 = RectangleSelector(ax3, onselect_axial,
                               useblit=True, button=[1],
                               minspanx=5, minspany=5,
                               spancoords='pixels',
                               interactive=True)
        
        # 连接滑块事件
        slider1.on_changed(update_sagittal)
        slider2.on_changed(update_coronal)
        slider3.on_changed(update_axial)
        
        # 显示数据范围信息
        # 显示使用说明
        usage_text = "Usage: Drag mouse on any view to select ROI region. Statistics will be displayed below."
        fig.text(0.5, 0.12, usage_text, ha='center', fontsize=10,
                style='italic', color='blue', weight='bold')
        
        info_text = (f"Data Range: [{self.data.min():.2f}, {self.data.max():.2f}] | "
                    f"Shape: {self.data.shape} | "
                    f"Voxel Size: {self.header.get_zooms()[0]:.2f}x{self.header.get_zooms()[1]:.2f}x{self.header.get_zooms()[2]:.2f}mm^3")
        fig.text(0.5, 0.01, info_text, ha='center', fontsize=9, style='italic')
        
        plt.show()
    
    def show_montage(self, num_slices: int = 16, view: str = 'axial'):
        """
        显示蒙太奇视图 (多个切片平铺显示)
        
        参数:
            num_slices: 显示的切片数量
            view: 视图类型 ('axial', 'coronal', 'sagittal')
        """
        if self.data is None:
            self.load_nifti()
        
        x_dim, y_dim, z_dim = self.data.shape
        
        # 根据视图类型选择切片
        if view == 'axial':
            total_slices = z_dim
            slice_name = 'Z axis (Axial)'
        elif view == 'coronal':
            total_slices = y_dim
            slice_name = 'Y axis (Coronal)'
        elif view == 'sagittal':
            total_slices = x_dim
            slice_name = 'X axis (Sagittal)'
        else:
            raise ValueError(f"Unsupported view type: {view}")
        
        # 确定显示的切片索引
        num_slices = min(num_slices, total_slices)
        slice_indices = np.linspace(0, total_slices-1, num_slices, dtype=int)
        
        # 计算子图网格
        cols = int(np.ceil(np.sqrt(num_slices)))
        rows = int(np.ceil(num_slices / cols))
        
        # 创建图形
        fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
        fig.suptitle(f'Montage View - {slice_name} - {os.path.basename(self.nifti_file)}',
                     fontsize=14, fontweight='bold')
        
        # 展平axes数组以便迭代
        if num_slices == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # 显示每个切片
        for idx, slice_idx in enumerate(slice_indices):
            if view == 'axial':
                slice_data = self.data[:, :, slice_idx]
            elif view == 'coronal':
                slice_data = self.data[:, slice_idx, :]
            else:  # sagittal
                slice_data = self.data[slice_idx, :, :]
            
            axes[idx].imshow(np.rot90(slice_data), cmap='gray', aspect='auto')
            axes[idx].set_title(f'{slice_name.split()[0]} = {slice_idx}', fontsize=9)
            axes[idx].axis('off')
        
        # 隐藏多余的子图
        for idx in range(num_slices, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='NIFTI文件查看器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 交互式查看
  python shownii.py -i output.nii.gz
  
  # 蒙太奇视图 (轴向面)
  python shownii.py -i output.nii.gz --montage
  
  # 蒙太奇视图 (冠状面，显示20个切片)
  python shownii.py -i output.nii.gz --montage --view coronal --num-slices 20
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='NIFTI文件路径'
    )
    parser.add_argument(
        '--montage',
        action='store_true',
        help='显示蒙太奇视图(多个切片平铺)'
    )
    parser.add_argument(
        '--view',
        choices=['axial', 'coronal', 'sagittal'],
        default='axial',
        help='蒙太奇视图的方向 (默认: axial)'
    )
    parser.add_argument(
        '--num-slices',
        type=int,
        default=16,
        help='蒙太奇视图显示的切片数量 (默认: 16)'
    )
    
    args = parser.parse_args()
    
    try:
        # 创建查看器
        viewer = NiftiViewer(args.input)
        
        # 显示视图
        if args.montage:
            viewer.show_montage(args.num_slices, args.view)
        else:
            viewer.show_interactive()
            
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()