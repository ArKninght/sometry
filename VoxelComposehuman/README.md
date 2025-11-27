# CT数据组合工具

这是一个用于将分开的CT体块数据（.raw格式）重新组合成完整多体块数据的Python工具。
支持3D体块数据（如256×256×64）的组合。

## 功能特性

- 支持3D体块数据（宽×高×深度，如256×256×64）
- 支持按体块分文件夹的数据结构（如 volume_1, volume_2, ...）
- 支持多种数据类型（uint8, uint16, uint32, float32, float64等）
- 两种组合模式：
  - **sequential**: 将所有体块组合到单个文件（4D数据：体块数×深度×高度×宽度）
  - **separate**: 保持体块分离，复制到新位置
- 自动验证文件完整性和体素数量
- 详细的处理日志

## 安装依赖

```bash
pip install numpy pyyaml
```

## 使用方法

### 1. 配置文件

首先编辑 `datainfo.yaml` 配置文件，设置您的数据参数：

```yaml
# 输入配置
input:
  folder: "./split_output"          # 输入目录
  structure: "frames"               # 文件夹结构类型
  num_frames: 10                    # 体块数量
  file_pattern: "*_frame_*.raw"     # 文件名模式

# 输出配置
output:
  folder: "./composed_output"       # 输出目录
  prefix: "composed"                # 输出文件名前缀
  overwrite: false                  # 是否覆盖已存在文件

# 数据参数
data:
  dtype: "uint16"                   # 数据类型
  width: 256                        # 体块宽度
  height: 256                       # 体块高度
  depth: 64                         # 体块深度
  byteorder: "little"               # 字节序

# 组合选项
compose:
  mode: "sequential"                # 组合模式
  sort_by: "name"                   # 排序方式
  reverse: false                    # 是否反转排序
```

### 2. 运行程序

#### 基本用法

```bash
python compose.py
```

#### 使用自定义配置文件

```bash
python compose.py -c myconfig.yaml
```

#### 组合并验证结果

```bash
python compose.py -v
```

### 3. 文件夹结构示例

#### 输入结构（frames模式）

```
split_output/
├── image_1/
│   ├── data001_frame_1.raw  (256×256×64体块)
│   ├── data002_frame_1.raw
│   └── ...
├── image_2/
│   ├── data001_frame_2.raw  (256×256×64体块)
│   ├── data002_frame_2.raw
│   └── ...
└── ...
```

#### 输出结构（sequential模式）

```
composed_output/
├── composed_data001.raw  (包含所有体块的4D数据)
├── composed_data002.raw
└── ...
```

每个输出文件的形状为：`[体块数, 64, 256, 256]`

#### 输出结构（separate模式）

```
composed_output/
├── frame_1/
│   ├── data001.raw  (单个256×256×64体块)
│   ├── data002.raw
│   └── ...
├── frame_2/
│   ├── data001.raw
│   ├── data002.raw
│   └── ...
└── ...
```

## 数据类型支持

| 类型 | 说明 | 字节数 |
|------|------|--------|
| uint8 | 无符号8位整数 | 1 |
| uint16 | 无符号16位整数 | 2 |
| uint32 | 无符号32位整数 | 4 |
| int16 | 有符号16位整数 | 2 |
| int32 | 有符号32位整数 | 4 |
| float32 | 32位浮点数 | 4 |
| float64 | 64位浮点数 | 8 |

## 组合模式说明

### Sequential模式

将同一组的所有体块按顺序组合到单个.raw文件中。适用于：
- 需要完整的多体块数据文件
- 后续处理需要访问所有体块
- 节省文件数量
- 需要4D数据格式

**输出文件格式**：`[体块数, 深度, 高度, 宽度]` 的4D数组
例如：10个体块，每个256×256×64 → 输出形状为 `[10, 64, 256, 256]`

### Separate模式

保持体块分离，只是重新组织文件结构。适用于：
- 需要单独访问每个体块
- 保持数据的模块化
- 便于按体块处理
- 每个体块保持原始的3D格式 `[深度, 高度, 宽度]`

## 验证功能

使用 `-v` 参数可以在组合完成后自动验证：
- 检查文件大小是否正确（验证体素总数）
- 显示数据形状（4D或3D）和数值范围
- 验证每个体块的尺寸是否为 256×256×64
- 列出所有输出文件

## 错误处理

程序会自动：
- 检查配置文件完整性
- 验证输入文件存在性
- 过滤不完整的文件组（缺少某些帧）
- 报告处理过程中的错误

## 示例

### 示例1：组合10个CT体块数据

```bash
# 编辑配置文件
vim datainfo.yaml

# 设置参数
# num_frames: 10
# dtype: "uint16"
# width: 256
# height: 256
# depth: 64

# 运行组合
python compose.py -v
```

输出将是10个体块组合成的4D数据，形状为 `[10, 64, 256, 256]`

### 示例2：从multirawsplite输出重组

如果您使用了 `multirawsplite/test.py` 分割数据，可以这样组合回去：

```yaml
input:
  folder: "/path/to/split_output"
  structure: "frames"
  num_frames: 10
  file_pattern: "*_frame_*.raw"

compose:
  mode: "sequential"
```

然后运行：
```bash
python compose.py -v
```

## 注意事项

1. **数据类型匹配**：确保配置文件中的dtype与实际数据类型一致
2. **尺寸正确**：width、height、depth必须与实际体块尺寸匹配（如256×256×64）
3. **文件完整性**：程序会自动跳过不完整的文件组
4. **磁盘空间**：sequential模式会创建较大的4D文件，确保有足够空间
   - 单个体块：256×256×64×2字节(uint16) ≈ 8.4 MB
   - 10个体块组合：≈ 84 MB
5. **覆盖保护**：默认不覆盖已存在文件，可通过配置修改
6. **内存使用**：组合多个体块时需要足够的内存来存储4D数组

## 配合使用

本工具可以与 [`multirawsplite/test.py`](../multirawsplite/test.py) 配合使用：

1. 使用 `test.py` 将多帧.raw文件分割成单帧
2. 进行图像处理或分析
3. 使用本工具将处理后的帧重新组合

## 故障排除

### 问题：文件大小不匹配

**解决**：检查配置文件中的数据类型、宽度和高度是否正确

### 问题：找不到文件

**解决**：
- 检查输入目录路径
- 确认file_pattern匹配实际文件名
- 查看文件夹结构是否符合预期

### 问题：内存不足

**解决**：
- 减少一次处理的文件数量
- 使用separate模式而非sequential模式
- 增加系统可用内存

## 作者

此工具与 `multirawsplite/test.py` 配套使用，用于CT数据的分割和组合处理。

## 许可

MIT License