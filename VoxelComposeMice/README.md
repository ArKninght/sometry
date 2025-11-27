# DICOM to NIFTI 转换工具

将一组CT重建的DICOM文件转换为NIFTI格式的Python工具。

## 功能特点

- 自动识别和排序DICOM文件
- 按切片位置正确排序
- 自动应用Rescale Slope和Intercept转换为HU值
- 提取并保存体素间距信息
- 支持压缩的NIFTI格式 (.nii.gz)
- 详细的处理日志和错误处理

## 依赖项

```bash
pip install pydicom nibabel numpy
```

## 使用方法

### 基本用法

```bash
python dcmcompose.py -i /path/to/dicom/folder
```

### 启用交互式ROI选择

```bash
python dcmcompose.py -i /path/to/dicom/folder --roi
```

### 指定输出文件名

```bash
python dcmcompose.py -i /path/to/dicom/folder -o my_scan.nii.gz
```

### 指定输出目录

```bash
python dcmcompose.py -i /path/to/dicom/folder -d ./output
```

### 完整示例(带ROI选择)

```bash
python dcmcompose.py \
    -i ./dicom_data \
    -o patient_001_ct.nii.gz \
    -d ./nifti_output \
    --roi
```

## 命令行参数

- `-i, --input`: (必需) DICOM文件所在文件夹路径
- `-o, --output`: 输出NIFTI文件名 (默认: output.nii.gz)
- `-d, --output-dir`: 输出文件夹路径 (默认: ./nifti_output)
- `--roi`: 启用交互式ROI(感兴趣区域)选择

## 程序说明

### 工作流程

1. **收集DICOM文件**: 自动扫描指定文件夹中的所有DICOM文件
2. **排序**: 根据ImagePositionPatient、SliceLocation或InstanceNumber排序
3. **读取数据**: 读取所有切片并应用HU值转换
4. **ROI选择**(可选): 在三个正交视图上交互式选择感兴趣区域
5. **裁剪**: 根据选择的ROI裁剪体块
6. **创建NIFTI**: 生成包含正确体素间距的NIFTI文件

### 核心类

#### `DicomToNiftiConverter`

主要转换类，包含以下方法：

- `collect_dicom_files()`: 收集和排序DICOM文件
- `read_dicom_series()`: 读取整个DICOM序列
- `select_roi_interactive()`: 交互式选择ROI区域
- `crop_volume()`: 根据ROI裁剪体块
- `save_as_nifti()`: 保存为NIFTI格式
- `convert()`: 执行完整转换流程

### 输出信息

程序会输出以下信息：

```
找到 256 个DICOM文件
成功排序 256 个DICOM文件

读取DICOM序列...
图像尺寸: 512 × 512 × 256
  已读取 50/256 个切片
  已读取 100/256 个切片
  ...
完成读取，数据范围: [-1024.00, 3071.00] HU

保存为NIFTI格式...
✓ 已保存: ./nifti_output/output.nii.gz
  形状: 512×512×256 (x×y×z)
  体素大小: 0.488 × 0.488 × 0.625 mm³
  数据类型: float32
  数据范围: [-1024.00, 3071.00] HU

转换完成!
```

## ROI选择功能

### 功能说明

启用`--roi`参数后,程序会在读取DICOM数据后显示一个交互式窗口,包含三个正交视图:

1. **轴向视图(Axial)**: XY平面,用于选择X和Y范围
2. **冠状视图(Coronal)**: XZ平面,用于选择X和Z范围
3. **矢状视图(Sagittal)**: YZ平面,用于选择Y和Z范围

### 操作方法

1. 在任意视图上**拖动鼠标**绘制矩形选择区域
2. 可以在不同视图上多次选择以精确定位ROI
3. 最后的选择会覆盖之前的选择
4. **关闭窗口**完成选择并继续转换

### 输出信息

程序会显示选择的ROI范围:

```
=== 交互式ROI选择 ===
操作说明:
  - 在三个视图上拖动鼠标选择感兴趣区域
  - 关闭窗口完成选择
  - 如果不选择,将使用完整体块

轴向视图选择: X=[100, 400], Y=[120, 380]
冠状视图选择: X=[100, 400], Z=[50, 200]
矢状视图选择: Y=[120, 380], Z=[50, 200]

最终ROI选择:
  X范围: 100 - 400 (共 300 像素)
  Y范围: 120 - 380 (共 260 像素)
  Z范围: 50 - 200 (共 150 切片)

裁剪完成:
  原始形状: (256, 512, 512)
  裁剪后形状: (150, 260, 300)
  数据范围: [-800.00, 2500.00] HU
```

## 支持的DICOM标签

程序会自动提取和使用以下DICOM标签：

- **图像尺寸**: Rows, Columns
- **切片位置**: ImagePositionPatient, SliceLocation, InstanceNumber
- **体素间距**: PixelSpacing, SliceThickness
- **HU转换**: RescaleSlope, RescaleIntercept
- **元数据**: PatientName, PatientID, SeriesDescription, StudyDate

## 坐标系统

输出的NIFTI文件使用RAS+坐标系统：
- R (Right): x轴正方向指向右侧
- A (Anterior): y轴正方向指向前方
- S (Superior): z轴正方向指向头顶

## 数据类型

- 输入: DICOM像素数组 (通常为int16)
- 处理: float32 (应用HU转换后)
- 输出: float32 NIFTI文件

## 错误处理

程序包含完善的错误处理机制：

1. 检查输入文件夹是否存在
2. 验证DICOM文件格式
3. 处理缺失的DICOM标签
4. 报告读取失败的切片

## 注意事项

1. **文件排序**: 程序会尝试多种方法对切片排序，确保正确的3D体数据
2. **HU值转换**: 自动应用Rescale Slope和Intercept，输出为标准HU值
3. **内存使用**: 大型CT数据集可能占用较多内存
4. **文件格式**: 默认生成压缩的.nii.gz文件以节省空间

## 示例代码

### 在Python脚本中使用

```python
from dcmcompose import DicomToNiftiConverter

# 创建转换器
converter = DicomToNiftiConverter(
    dicom_folder="./dicom_data",
    output_folder="./nifti_output"
)

# 执行转换(不启用ROI选择)
output_path = converter.convert("output.nii.gz")
print(f"转换完成: {output_path}")

# 或启用ROI选择
output_path = converter.convert("output.nii.gz", enable_roi_selection=True)
print(f"转换完成: {output_path}")
```

### 批量转换多个序列

```python
import os
from dcmcompose import DicomToNiftiConverter

# 定义多个DICOM序列
series_folders = [
    "./patient_001/series_01",
    "./patient_001/series_02",
    "./patient_002/series_01"
]

# 逐个转换
for folder in series_folders:
    if os.path.exists(folder):
        converter = DicomToNiftiConverter(folder, "./nifti_output")
        output_name = f"{os.path.basename(folder)}.nii.gz"
        converter.convert(output_name)
```

## 疑难解答

### 问题：找不到DICOM文件

**解决方案**: 
- 确认文件夹路径正确
- 确认文件是有效的DICOM格式
- 程序会尝试读取不带.dcm扩展名的文件

### 问题：切片顺序错误

**解决方案**: 
- 检查DICOM文件是否包含ImagePositionPatient或SliceLocation标签
- 程序会按这些标签自动排序

### 问题：体素间距不正确

**解决方案**: 
- 检查DICOM文件的PixelSpacing和SliceThickness标签
- 如果缺失，程序会使用默认值1.0mm

## 版本信息

- 版本: 1.0
- 作者: VoxelComposeMice
- 更新日期: 2024

## 许可证

MIT License