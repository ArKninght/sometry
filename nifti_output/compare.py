import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import gzip

def compare_nifti_files(file1_path, file2_path):
    """
    比较两个 NIfTI 文件的详细信息和数据差异
    
    参数:
        file1_path: 第一个 NIfTI 文件路径
        file2_path: 第二个 NIfTI 文件路径
    """
    print("=" * 80)
    print("NIfTI 文件比较分析")
    print("=" * 80)
    
    # 文件大小分析
    print("\n" + "=" * 80)
    print("0. 文件大小分析")
    print("=" * 80)
    
    file1_size = os.path.getsize(file1_path)
    file2_size = os.path.getsize(file2_path)
    
    print(f"\n文件1: {file1_path}")
    print(f"  文件大小: {file1_size / (1024*1024):.2f} MB ({file1_size:,} bytes)")
    
    print(f"\n文件2: {file2_path}")
    print(f"  文件大小: {file2_size / (1024*1024):.2f} MB ({file2_size:,} bytes)")
    
    print(f"\n大小差异: {abs(file2_size - file1_size) / (1024*1024):.2f} MB")
    print(f"大小比例: {max(file1_size, file2_size) / min(file1_size, file2_size):.2f}x")
    
    # 加载文件
    print(f"\n加载文件内容:")
    
    try:
        nifti1 = nib.load(file1_path)
        nifti2 = nib.load(file2_path)
        
        data1 = nifti1.get_fdata()
        data2 = nifti2.get_fdata()
        
        header1 = nifti1.header
        header2 = nifti2.header
        
    except Exception as e:
        print(f"\n错误: 无法加载文件 - {e}")
        return
    
    # 1. 基本信息比较
    print("\n" + "=" * 80)
    print("1. 基本信息比较")
    print("=" * 80)
    
    print(f"\n{'属性':<30} {'文件1 (LungCT_0001)':<25} {'文件2 (output_128)':<25}")
    print("-" * 80)
    print(f"{'数据形状':<30} {str(data1.shape):<25} {str(data2.shape):<25}")
    print(f"{'数据类型':<30} {str(data1.dtype):<25} {str(data2.dtype):<25}")
    print(f"{'维度数':<30} {data1.ndim:<25} {data2.ndim:<25}")
    print(f"{'总体素数':<30} {data1.size:<25} {data2.size:<25}")
    
    # 计算理论内存大小
    theoretical_size1 = data1.nbytes
    theoretical_size2 = data2.nbytes
    print(f"{'理论数据大小(MB)':<30} {theoretical_size1/(1024*1024):<25.2f} {theoretical_size2/(1024*1024):<25.2f}")
    
    # 压缩率分析
    compression_ratio1 = file1_size / theoretical_size1
    compression_ratio2 = file2_size / theoretical_size2
    print(f"{'压缩率':<30} {compression_ratio1:<25.2%} {compression_ratio2:<25.2%}")
    print(f"{'压缩效率':<30} {(1-compression_ratio1):<25.2%} {(1-compression_ratio2):<25.2%}")
    
    # 2. 体素尺寸和物理空间信息
    print("\n" + "=" * 80)
    print("2. 体素尺寸和空间分辨率")
    print("=" * 80)
    
    pixdim1 = header1['pixdim'][1:4]
    pixdim2 = header2['pixdim'][1:4]
    
    print(f"\n{'体素尺寸 (mm)':<30} {str(pixdim1):<25} {str(pixdim2):<25}")
    print(f"{'物理尺寸 X (mm)':<30} {data1.shape[0]*pixdim1[0]:<25.2f} {data2.shape[0]*pixdim2[0]:<25.2f}")
    print(f"{'物理尺寸 Y (mm)':<30} {data1.shape[1]*pixdim1[1]:<25.2f} {data2.shape[1]*pixdim2[1]:<25.2f}")
    print(f"{'物理尺寸 Z (mm)':<30} {data1.shape[2]*pixdim1[2]:<25.2f} {data2.shape[2]*pixdim2[2]:<25.2f}")
    
    # 3. 数据统计信息
    print("\n" + "=" * 80)
    print("3. 数据统计信息")
    print("=" * 80)
    
    print(f"\n{'统计量':<30} {'文件1 (LungCT_0001)':<25} {'文件2 (output_128)':<25}")
    print("-" * 80)
    print(f"{'最小值':<30} {np.min(data1):<25.2f} {np.min(data2):<25.2f}")
    print(f"{'最大值':<30} {np.max(data1):<25.2f} {np.max(data2):<25.2f}")
    print(f"{'平均值':<30} {np.mean(data1):<25.2f} {np.mean(data2):<25.2f}")
    print(f"{'标准差':<30} {np.std(data1):<25.2f} {np.std(data2):<25.2f}")
    print(f"{'中位数':<30} {np.median(data1):<25.2f} {np.median(data2):<25.2f}")
    
    # 4. 数据差异分析
    print("\n" + "=" * 80)
    print("4. 数据差异分析")
    print("=" * 80)
    
    if data1.shape == data2.shape:
        diff = data2 - data1
        print(f"\n两个文件具有相同的形状，可以进行逐体素比较")
        print(f"差异统计:")
        print(f"  - 平均差异: {np.mean(diff):.4f}")
        print(f"  - 最大差异: {np.max(np.abs(diff)):.4f}")
        print(f"  - 差异标准差: {np.std(diff):.4f}")
        print(f"  - 相同体素数: {np.sum(np.abs(diff) < 1e-6)}")
        print(f"  - 不同体素数: {np.sum(np.abs(diff) >= 1e-6)}")
        print(f"  - 相似度百分比: {(np.sum(np.abs(diff) < 1e-6) / diff.size * 100):.2f}%")
        
        # 5. 可视化比较
        print("\n生成可视化对比图...")
        visualize_comparison(data1, data2, diff)
        
    else:
        print(f"\n两个文件形状不同:")
        print(f"  - 文件1形状: {data1.shape}")
        print(f"  - 文件2形状: {data2.shape}")
        print(f"  - 形状比例: {np.array(data1.shape) / np.array(data2.shape)}")
        print(f"\n这可能表明 output_128 是 LungCT_0001 的下采样版本")
        
        # 生成单独的可视化
        print("\n生成独立可视化图...")
        visualize_separate(data1, data2)
    
    # 6. 头文件信息比较
    print("\n" + "=" * 80)
    print("5. 头文件关键信息")
    print("=" * 80)
    
    # NIfTI 数据类型编码映射
    nifti_datatype_map = {
        0: 'UNKNOWN (未知)',
        1: 'BINARY (二进制/1位)',
        2: 'UINT8 (无符号8位整数)',
        4: 'INT16 (有符号16位整数)',
        8: 'INT32 (有符号32位整数)',
        16: 'FLOAT32 (32位浮点数)',
        32: 'COMPLEX64 (64位复数)',
        64: 'FLOAT64 (64位浮点数/双精度)',
        128: 'RGB24 (24位RGB)',
        255: 'ALL (所有类型)',
        256: 'INT8 (有符号8位整数)',
        512: 'UINT16 (无符号16位整数)',
        768: 'UINT32 (无符号32位整数)',
        1024: 'INT64 (有符号64位整数)',
        1280: 'UINT64 (无符号64位整数)',
        1536: 'FLOAT128 (128位浮点数)',
        1792: 'COMPLEX128 (128位复数)',
        2048: 'COMPLEX256 (256位复数)',
    }
    
    datatype1_code = int(header1['datatype'])
    datatype2_code = int(header2['datatype'])
    
    datatype1_name = nifti_datatype_map.get(datatype1_code, f'未知类型({datatype1_code})')
    datatype2_name = nifti_datatype_map.get(datatype2_code, f'未知类型({datatype2_code})')
    
    print(f"\n文件1 - 数据类型:")
    print(f"  编码: {datatype1_code}")
    print(f"  类型: {datatype1_name}")
    print(f"  实际Python类型: {data1.dtype}")
    print(f"  每个体素字节数: {data1.dtype.itemsize} bytes")
    
    print(f"\nFile2 - 数据类型:")
    print(f"  编码: {datatype2_code}")
    print(f"  类型: {datatype2_name}")
    print(f"  实际Python类型: {data2.dtype}")
    print(f"  每个体素字节数: {data2.dtype.itemsize} bytes")
    
    if datatype1_code != datatype2_code:
        print(f"\n⚠️  注意: 两个文件使用不同的数据类型!")
        print(f"  这是导致文件大小差异的主要原因之一")
        bytes_diff = abs(data1.dtype.itemsize - data2.dtype.itemsize) * data1.size
        print(f"  仅数据类型差异就会导致约 {bytes_diff/(1024*1024):.2f} MB 的理论大小差异")
    
    # 7. 文件大小差异原因分析
    print("\n" + "=" * 80)
    print("6. 文件大小差异原因分析")
    print("=" * 80)
    
    print(f"\n文件大小差异的可能原因:")
    
    # 原因1: 数据类型差异
    if data1.dtype != data2.dtype:
        print(f"\n✓ 原因1: 数据类型不同")
        print(f"  文件1使用 {data1.dtype}, 每个体素占 {data1.dtype.itemsize} 字节")
        print(f"  文件2使用 {data2.dtype}, 每个体素占 {data2.dtype.itemsize} 字节")
        print(f"  这会导致理论数据大小相差 {abs(theoretical_size1-theoretical_size2)/(1024*1024):.2f} MB")
    else:
        print(f"\n✗ 数据类型相同 ({data1.dtype}), 不是导致大小差异的原因")
    
    # 原因2: 压缩效率
    print(f"\n✓ 原因2: 压缩效率差异 (最可能的原因)")
    print(f"  文件1压缩率: {compression_ratio1:.2%} (压缩后占原始数据的{compression_ratio1:.2%})")
    print(f"  文件2压缩率: {compression_ratio2:.2%} (压缩后占原始数据的{compression_ratio2:.2%})")
    
    # 分析数据特性对压缩的影响
    # 计算数据的"可压缩性"指标
    
    # 1. 零值比例
    zero_ratio1 = np.sum(np.abs(data1) < 1e-6) / data1.size
    zero_ratio2 = np.sum(np.abs(data2) < 1e-6) / data2.size
    
    # 2. 数据重复性 (通过唯一值比例估算)
    unique_ratio1 = len(np.unique(data1)) / data1.size
    unique_ratio2 = len(np.unique(data2)) / data2.size
    
    # 3. 数据变化率 (通过标准差估算)
    normalized_std1 = np.std(data1) / (np.max(data1) - np.min(data1) + 1e-10)
    normalized_std2 = np.std(data2) / (np.max(data2) - np.min(data2) + 1e-10)
    
    print(f"\n  数据可压缩性分析:")
    print(f"  {'指标':<25} {'文件1':<20} {'文件2':<20}")
    print(f"  {'-'*65}")
    print(f"  {'零值/空白区域占比':<25} {zero_ratio1:<20.2%} {zero_ratio2:<20.2%}")
    print(f"  {'唯一值占比':<25} {unique_ratio1:<20.2%} {unique_ratio2:<20.2%}")
    print(f"  {'归一化标准差':<25} {normalized_std1:<20.4f} {normalized_std2:<20.4f}")
    
    print(f"\n  解释:")
    print(f"  - 零值占比越高,压缩效果越好 (更多空白区域易压缩)")
    print(f"  - 唯一值占比越低,压缩效果越好 (更多重复数据)")
    print(f"  - 归一化标准差越低,压缩效果越好 (数据变化更平缓)")
    
    # 总结
    print(f"\n结论:")
    if compression_ratio1 < compression_ratio2:
        better = "文件1"
        worse = "文件2"
    else:
        better = "文件2"
        worse = "文件1"
    
    print(f"  {better} 的压缩效果更好,可能因为:")
    if zero_ratio1 > zero_ratio2:
        print(f"    - 文件1 包含更多空白/零值区域 ({zero_ratio1:.1%} vs {zero_ratio2:.1%})")
    elif zero_ratio2 > zero_ratio1:
        print(f"    - 文件2 包含更多空白/零值区域 ({zero_ratio2:.1%} vs {zero_ratio1:.1%})")
    
    if unique_ratio1 < unique_ratio2:
        print(f"    - 文件1 数据重复性更高 (唯一值占比: {unique_ratio1:.1%} vs {unique_ratio2:.1%})")
    elif unique_ratio2 < unique_ratio1:
        print(f"    - 文件2 数据重复性更高 (唯一值占比: {unique_ratio2:.1%} vs {unique_ratio1:.1%})")
    
    print(f"\n  实际文件大小差异 {abs(file2_size - file1_size)/(1024*1024):.2f} MB 主要由压缩效率不同造成")
    
    print("\n" + "=" * 80)
    print("比较完成！")
    print("=" * 80)


def visualize_comparison(data1, data2, diff):
    """
    可视化两个相同形状数据的比较
    """
    # 选择中间切片
    slice_idx = data1.shape[2] // 2
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('NIfTI 文件比较 (中间切片)', fontsize=16, fontweight='bold')
    
    # 第一行 - 轴向视图
    im1 = axes[0, 0].imshow(data1[:, :, slice_idx].T, cmap='gray', origin='lower')
    axes[0, 0].set_title('LungCT_0001 - 轴向')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
    
    im2 = axes[0, 1].imshow(data2[:, :, slice_idx].T, cmap='gray', origin='lower')
    axes[0, 1].set_title('output_128 - 轴向')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
    
    im3 = axes[0, 2].imshow(diff[:, :, slice_idx].T, cmap='RdBu_r', origin='lower')
    axes[0, 2].set_title('差异图 - 轴向')
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)
    
    # 第二行 - 矢状视图
    slice_idx_sag = data1.shape[0] // 2
    im4 = axes[1, 0].imshow(data1[slice_idx_sag, :, :].T, cmap='gray', origin='lower')
    axes[1, 0].set_title('LungCT_0001 - 矢状')
    axes[1, 0].axis('off')
    plt.colorbar(im4, ax=axes[1, 0], fraction=0.046)
    
    im5 = axes[1, 1].imshow(data2[slice_idx_sag, :, :].T, cmap='gray', origin='lower')
    axes[1, 1].set_title('output_128 - 矢状')
    axes[1, 1].axis('off')
    plt.colorbar(im5, ax=axes[1, 1], fraction=0.046)
    
    im6 = axes[1, 2].imshow(diff[slice_idx_sag, :, :].T, cmap='RdBu_r', origin='lower')
    axes[1, 2].set_title('差异图 - 矢状')
    axes[1, 2].axis('off')
    plt.colorbar(im6, ax=axes[1, 2], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig('nifti_output/comparison_same_shape.png', dpi=150, bbox_inches='tight')
    print(f"  - 保存对比图到: nifti_output/comparison_same_shape.png")
    plt.close()


def visualize_separate(data1, data2):
    """
    可视化两个不同形状数据的独立视图
    """
    slice_idx1 = data1.shape[2] // 2
    slice_idx2 = data2.shape[2] // 2
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle('NIfTI 文件独立比较', fontsize=16, fontweight='bold')
    
    # LungCT_0001
    im1 = axes[0, 0].imshow(data1[:, :, slice_idx1].T, cmap='gray', origin='lower')
    axes[0, 0].set_title(f'LungCT_0001 - 轴向 (切片 {slice_idx1})\n形状: {data1.shape}')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
    
    slice_idx1_sag = data1.shape[0] // 2
    im2 = axes[0, 1].imshow(data1[slice_idx1_sag, :, :].T, cmap='gray', origin='lower')
    axes[0, 1].set_title(f'LungCT_0001 - 矢状 (切片 {slice_idx1_sag})')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
    
    # output_128
    im3 = axes[1, 0].imshow(data2[:, :, slice_idx2].T, cmap='gray', origin='lower')
    axes[1, 0].set_title(f'output_128 - 轴向 (切片 {slice_idx2})\n形状: {data2.shape}')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
    
    slice_idx2_sag = data2.shape[0] // 2
    im4 = axes[1, 1].imshow(data2[slice_idx2_sag, :, :].T, cmap='gray', origin='lower')
    axes[1, 1].set_title(f'output_128 - 矢状 (切片 {slice_idx2_sag})')
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig('nifti_output/comparison_different_shapes.png', dpi=150, bbox_inches='tight')
    print(f"  - 保存对比图到: nifti_output/comparison_different_shapes.png")
    plt.close()


if __name__ == "__main__":
    # 定义文件路径
    file1 = "nifti_output/LungCT_0001_0005.nii.gz"
    file2 = "nifti_output/output_128.nii.gz"
    
    # 检查文件是否存在
    if not Path(file1).exists():
        print(f"错误: 找不到文件 {file1}")
        exit(1)
    
    if not Path(file2).exists():
        print(f"错误: 找不到文件 {file2}")
        exit(1)
    
    # 执行比较
    compare_nifti_files(file1, file2)