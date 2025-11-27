#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4D CT NIFTI查看器
交互式查看4D CT序列（3D空间 + 时间维度）
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import argparse
import os


class FourDCTViewer:
    """4D CT查看器"""
    
    def __init__(self, nifti_file: str):
        """
        初始化查看器
        
        参数:
            nifti_file: 4D NIFTI文件路径
        """
        if not os.path.exists(nifti_file):
            raise FileNotFoundError(f"文件不存在: {nifti_file}")
        
        self.nifti_file = nifti_file
        self.img = None
        self.data = None
        self.affine = None
        self.header = None
        
    def load_nifti(self):
        """加载4D NIFTI文件"""
        print(f"加载文件: {self.nifti_file}")
        
        # 读取NIFTI文件
        self.img = nib.load(self.nifti_file)
        self.data = self.img.get_fdata()
        self.affine = self.img.affine
        self.header = self.img.header
        
        # 显示基本信息
        print(f"\n文件信息:")
        print(f"  形状: {self.data.shape}")
        print(f"  数据类型: {self.data.dtype}")
        print(f"  数据范围: [{self.data.min():.2f}, {self.data.max():.2f}]")
        
        # 检查是否为4D数据
        if len(self.data.shape) != 4:
            raise ValueError(f"不是4D数据! 数据维度: {len(self.data.shape)}")
        
        # 体素尺寸
        voxel_sizes = self.header.get_zooms()
        print(f"  体素尺寸: {voxel_sizes[0]:.3f} x {voxel_sizes[1]:.3f} x {voxel_sizes[2]:.3f} mm^3")
        if len(voxel_sizes) > 3:
            print(f"  时间间隔: {voxel_sizes[3]:.3f} 秒")
        
        print(f"  时间帧数: {self.data.shape[3]}")
        
        return self.data
    
    def show_interactive(self):
        """交互式显示4D CT"""
        if self.data is None:
            self.load_nifti()
        
        # 数据维度 (x, y, z, t)
        x_dim, y_dim, z_dim, t_dim = self.data.shape
        
        # 初始位置
        x_slice = x_dim // 2
        y_slice = y_dim // 2
        z_slice = z_dim // 2
        t_frame = 0
        
        # 创建图形
        fig = plt.figure(figsize=(18, 6))
        fig.suptitle(f'4D CT Viewer - {os.path.basename(self.nifti_file)}',
                     fontsize=14, fontweight='bold')
        
        # 矢状面 (Sagittal) - YZ平面
        ax1 = plt.subplot(131)
        sagittal_data = self.data[x_slice, :, :, t_frame]
        im1 = ax1.imshow(np.rot90(sagittal_data), cmap='gray', aspect='auto',
                        vmin=self.data.min(), vmax=self.data.max())
        ax1.set_title(f'矢状面 (Sagittal)\nX={x_slice}, T={t_frame}', fontsize=11)
        ax1.set_xlabel('Z轴')
        ax1.set_ylabel('Y轴')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # 冠状面 (Coronal) - XZ平面
        ax2 = plt.subplot(132)
        coronal_data = self.data[:, y_slice, :, t_frame]
        im2 = ax2.imshow(np.rot90(coronal_data), cmap='gray', aspect='auto',
                        vmin=self.data.min(), vmax=self.data.max())
        ax2.set_title(f'冠状面 (Coronal)\nY={y_slice}, T={t_frame}', fontsize=11)
        ax2.set_xlabel('Z轴')
        ax2.set_ylabel('X轴')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        # 轴向面 (Axial) - XY平面
        ax3 = plt.subplot(133)
        axial_data = self.data[:, :, z_slice, t_frame]
        im3 = ax3.imshow(np.rot90(axial_data), cmap='gray', aspect='auto',
                        vmin=self.data.min(), vmax=self.data.max())
        ax3.set_title(f'轴向面 (Axial)\nZ={z_slice}, T={t_frame}', fontsize=11)
        ax3.set_xlabel('X轴')
        ax3.set_ylabel('Y轴')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        
        # 调整子图布局
        plt.subplots_adjust(left=0.05, right=0.95, top=0.88, bottom=0.30, wspace=0.3)
        
        # 创建滑块
        # 矢状面滑块 (X)
        ax_slider1 = plt.axes([0.12, 0.18, 0.15, 0.03])
        slider_x = Slider(ax_slider1, 'X (矢状)', 0, x_dim-1,
                         valinit=x_slice, valstep=1, color='#ff7f0e')
        
        # 冠状面滑块 (Y)
        ax_slider2 = plt.axes([0.42, 0.18, 0.15, 0.03])
        slider_y = Slider(ax_slider2, 'Y (冠状)', 0, y_dim-1,
                         valinit=y_slice, valstep=1, color='#2ca02c')
        
        # 轴向面滑块 (Z)
        ax_slider3 = plt.axes([0.72, 0.18, 0.15, 0.03])
        slider_z = Slider(ax_slider3, 'Z (轴向)', 0, z_dim-1,
                         valinit=z_slice, valstep=1, color='#d62728')
        
        # 时间滑块 (T)
        ax_slider4 = plt.axes([0.12, 0.12, 0.75, 0.03])
        slider_t = Slider(ax_slider4, '时间帧 (T)', 0, t_dim-1,
                         valinit=t_frame, valstep=1, color='#9467bd')
        
        # 播放控制按钮
        ax_play = plt.axes([0.42, 0.05, 0.08, 0.04])
        btn_play = Button(ax_play, '播放')
        
        ax_pause = plt.axes([0.51, 0.05, 0.08, 0.04])
        btn_pause = Button(ax_pause, '暂停')
        
        # 播放状态
        self.is_playing = False
        self.timer = None
        
        # 更新函数
        def update_views(x=None, y=None, z=None, t=None):
            """更新所有视图"""
            current_x = int(slider_x.val) if x is None else x
            current_y = int(slider_y.val) if y is None else y
            current_z = int(slider_z.val) if z is None else z
            current_t = int(slider_t.val) if t is None else t
            
            # 更新矢状面
            sagittal_data = self.data[current_x, :, :, current_t]
            im1.set_data(np.rot90(sagittal_data))
            ax1.set_title(f'矢状面 (Sagittal)\nX={current_x}, T={current_t}', fontsize=11)
            
            # 更新冠状面
            coronal_data = self.data[:, current_y, :, current_t]
            im2.set_data(np.rot90(coronal_data))
            ax2.set_title(f'冠状面 (Coronal)\nY={current_y}, T={current_t}', fontsize=11)
            
            # 更新轴向面
            axial_data = self.data[:, :, current_z, current_t]
            im3.set_data(np.rot90(axial_data))
            ax3.set_title(f'轴向面 (Axial)\nZ={current_z}, T={current_t}', fontsize=11)
            
            fig.canvas.draw_idle()
        
        # 滑块事件处理
        def on_slider_x(val):
            update_views(x=int(val))
        
        def on_slider_y(val):
            update_views(y=int(val))
        
        def on_slider_z(val):
            update_views(z=int(val))
        
        def on_slider_t(val):
            update_views(t=int(val))
        
        # 播放/暂停控制
        def play_animation(event):
            """播放动画"""
            self.is_playing = True
            
            def animate():
                if self.is_playing:
                    current_t = int(slider_t.val)
                    next_t = (current_t + 1) % t_dim
                    slider_t.set_val(next_t)
                    self.timer = fig.canvas.new_timer(interval=200)  # 200ms间隔
                    self.timer.single_shot = True
                    self.timer.add_callback(animate)
                    self.timer.start()
            
            animate()
        
        def pause_animation(event):
            """暂停动画"""
            self.is_playing = False
            if self.timer is not None:
                self.timer.stop()
        
        # 连接事件
        slider_x.on_changed(on_slider_x)
        slider_y.on_changed(on_slider_y)
        slider_z.on_changed(on_slider_z)
        slider_t.on_changed(on_slider_t)
        btn_play.on_clicked(play_animation)
        btn_pause.on_clicked(pause_animation)
        
        # 显示信息
        info_text = (f"形状: {self.data.shape} | "
                    f"数据范围: [{self.data.min():.2f}, {self.data.max():.2f}] | "
                    f"体素尺寸: {self.header.get_zooms()[0]:.2f}x{self.header.get_zooms()[1]:.2f}x{self.header.get_zooms()[2]:.2f}mm^3")
        fig.text(0.5, 0.01, info_text, ha='center', fontsize=9, style='italic')
        
        plt.show()
    
    def export_frames(self, output_dir: str = "./frames", stride: int = 1):
        """
        导出时间帧为单独的图像
        
        参数:
            output_dir: 输出目录
            stride: 导出间隔（每隔stride帧导出一次）
        """
        if self.data is None:
            self.load_nifti()
        
        os.makedirs(output_dir, exist_ok=True)
        
        x_dim, y_dim, z_dim, t_dim = self.data.shape
        z_mid = z_dim // 2
        
        print(f"\n导出时间帧到 {output_dir}...")
        
        for t in range(0, t_dim, stride):
            # 提取中间轴向切片
            axial_slice = self.data[:, :, z_mid, t]
            
            # 保存为图像
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(np.rot90(axial_slice), cmap='gray', aspect='auto')
            ax.set_title(f'时间帧 {t} / {t_dim-1}', fontsize=14, fontweight='bold')
            ax.axis('off')
            
            output_file = os.path.join(output_dir, f"frame_{t:03d}.png")
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  已保存: {output_file}")
        
        print(f"\n完成! 共导出 {len(range(0, t_dim, stride))} 帧")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='4D CT NIFTI查看器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 交互式查看
  python show4DCTnii.py -i output_4DCT_generated.nii.gz
  
  # 导出时间帧
  python show4DCTnii.py -i output_4DCT_generated.nii.gz --export --output-dir ./frames
  
  # 导出时间帧（每2帧导出一次）
  python show4DCTnii.py -i output_4DCT_generated.nii.gz --export --stride 2
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='4D NIFTI文件路径'
    )
    parser.add_argument(
        '--export',
        action='store_true',
        help='导出时间帧为图像'
    )
    parser.add_argument(
        '--output-dir',
        default='./frames',
        help='导出图像的输出目录 (默认: ./frames)'
    )
    parser.add_argument(
        '--stride',
        type=int,
        default=1,
        help='导出帧间隔 (默认: 1, 导出所有帧)'
    )
    
    args = parser.parse_args()
    
    try:
        # 创建查看器
        viewer = FourDCTViewer(args.input)
        
        if args.export:
            # 导出时间帧
            viewer.export_frames(args.output_dir, args.stride)
        else:
            # 交互式查看
            viewer.show_interactive()
            
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()