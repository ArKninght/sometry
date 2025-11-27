# crop_dcm_to_nifti.py 与 dcmcompose.py 的差异分析

## 问题描述
使用 `crop_dcm_to_nifti.py` 和 `dcmcompose.py` 从相同的DICOM体块中裁剪ROI，得到的NIFTI图像在X轴和Z轴上不同。

## 关键差异

### 1. **NIFTI保存时的维度转置差异**

#### crop_dcm_to_nifti.py (第186行)
```python
# 直接保存，没有转置
nifti_img = nib.Nifti1Image(volume, affine)
# volume 的形状是 (Z, Y, X)
```

#### dcmcompose.py (第747行)
```python
# 保存前进行了维度转置 (Z, Y, X) -> (X, Y, Z)
volume_transposed = np.transpose(volume, (2, 1, 0))
nifti_img = nib.Nifti1Image(volume_transposed, affine)
```

#### compose4Dandraw.py (第93-94行)
```python
# 加载NIFTI时进行了转置 (X, Y, Z) -> (Z, Y, X)
volume = img.get_fdata(dtype=np.float32)
volume = np.transpose(volume, (2, 1, 0))
```

### 2. **仿射矩阵的构建差异**

#### crop_dcm_to_nifti.py (第180-183行)
```python
affine = np.eye(4)
affine[0, 0] = pixel_spacing[0]  # 对应X轴
affine[1, 1] = pixel_spacing[1]  # 对应Y轴
affine[2, 2] = slice_thickness    # 对应Z轴
```

#### dcmcompose.py (第188-191行)
```python
affine = np.eye(4)
affine[0, 0] = pixel_spacing[1]  # 列间距 (x)
affine[1, 1] = pixel_spacing[0]  # 行间距 (y)
affine[2, 2] = slice_thickness    # 切片间距 (z)
```

**注意**: `pixel_spacing[0]` 和 `pixel_spacing[1]` 的赋值顺序相反！

### 3. **裁剪操作的坐标系统**

#### crop_dcm_to_nifti.py (第158行)
```python
# 直接按照 (Z, Y, X) 顺序裁剪
cropped = volume[z_min:z_max, y_min:y_max, x_min:x_max]
```

#### dcmcompose.py (第403行)
```python
# 同样按照 (Z, Y, X) 顺序裁剪
cropped = volume[z_min:z_max, y_min:y_max, x_min:x_max]
```

## 根本原因

**NIFTI格式的约定**: 
- NIFTI文件存储时，数据应该是 (X, Y, Z) 顺序
- 但在Python/NumPy中处理DICOM时，通常使用 (Z, Y, X) 顺序

**两种方法的不同处理方式**:

1. **dcmcompose.py**: 
   - 在内存中使用 (Z, Y, X) 进行处理
   - 保存NIFTI前转置为 (X, Y, Z)
   - 这是正确的做法

2. **crop_dcm_to_nifti.py**:
   - 在内存中使用 (Z, Y, X) 进行处理
   - 保存NIFTI时**没有转置**，直接保存 (Z, Y, X)
   - 导致X轴和Z轴互换

## 影响

由于 `crop_dcm_to_nifti.py` 没有进行维度转置:
- **Z轴数据** 被错误地存储为 **X轴**
- **X轴数据** 被错误地存储为 **Z轴**
- Y轴保持不变

这导致:
- 图像看起来是"旋转"或"翻转"的
- ROI坐标系统与实际数据不匹配
- 与 `compose4Dandraw.py` 的输出不兼容

## 解决方案

修改 `crop_dcm_to_nifti.py` 的 `save_as_nifti()` 方法，在保存前添加维度转置:

```python
def save_as_nifti(self, volume: np.ndarray, metadata: Dict, output_name: str = "cropped_volume.nii.gz"):
    print(f"\n保存为NIFTI格式...")
    
    # 维度转置: (Z, Y, X) -> (X, Y, Z) 以符合NIFTI标准
    volume_transposed = np.transpose(volume, (2, 1, 0))
    
    # 创建仿射矩阵
    pixel_spacing = metadata['PixelSpacing']
    slice_thickness = metadata['SliceThickness']
    
    affine = np.eye(4)
    affine[0, 0] = pixel_spacing[1]  # 列间距 (x)
    affine[1, 1] = pixel_spacing[0]  # 行间距 (y)
    affine[2, 2] = slice_thickness    # 切片间距 (z)
    
    # 使用转置后的数据创建NIFTI图像
    nifti_img = nib.Nifti1Image(volume_transposed, affine)
    
    # 保存
    output_path = os.path.join(self.output_folder, output_name)
    nib.save(nifti_img, output_path)
    
    print(f"  ✓ 已保存到: {output_path}")
    print(f"  文件大小: {os.path.getsize(output_path) / (1024**2):.2f} MB")
```

## 总结

关键问题在于:
1. NIFTI格式要求 (X, Y, Z) 存储顺序
2. `crop_dcm_to_nifti.py` 没有进行必要的维度转置
3. `dcmcompose.py` 正确地进行了转置操作
4. 仿射矩阵中 `pixel_spacing` 的索引顺序也需要相应调整