# 运动模式详解

本文档详细解释 [`mixed.py`](mixed.py:68) 中三种运动插入方法的工作原理。

## 核心概念

假设我们有 **5个phase** (Phase 0-4) 和 **360个投影角度** (0°-360°)。

目标: 为每个投影角度分配一个phase,模拟呼吸运动。

---

## 1. Sinusoidal (正弦运动) - 推荐

**公式** (代码第87-88行):
```python
t = np.linspace(0, 4*np.pi, num_angles)  # 时间参数: 0 到 4π (2个周期)
phase_continuous = (np.sin(t) + 1) / 2 * (num_phases - 1)
```

**工作原理**:
- 使用正弦函数模拟平滑的呼吸运动
- `sin(t)` 的值在 [-1, 1] 之间振荡
- `(sin(t) + 1) / 2` 归一化到 [0, 1]
- 乘以 `(num_phases - 1)` 得到phase索引

**示例** (5个phase, 360角度):
```
角度 0°:   phase ≈ 0 (呼气末)
角度 45°:  phase ≈ 2 (吸气中)
角度 90°:  phase ≈ 4 (吸气末)
角度 135°: phase ≈ 2 (呼气中)
角度 180°: phase ≈ 0 (呼气末)
...重复
```

**运动曲线**:
```
Phase
  4 |    ╱╲      ╱╲
  3 |   ╱  ╲    ╱  ╲
  2 |  ╱    ╲  ╱    ╲
  1 | ╱      ╲╱      ╲
  0 |╱                ╲
    +--------------------> 角度
    0°   90°  180° 270° 360°
```

**特点**:
- ✅ 平滑过渡,最接近真实呼吸
- ✅ 每个phase的使用频率不同(phase 0和4多,phase 2少)
- ✅ 适合模拟周期性呼吸运动

---

## 2. Linear (线性循环)

**公式** (代码第92行):
```python
phase_continuous = np.linspace(0, num_phases - 1, num_angles) % num_phases
```

**工作原理**:
- 线性递增phase索引
- 到达最大phase后,通过取模 `% num_phases` 回到0

**示例** (5个phase, 360角度):
```
角度 0°:   phase = 0
角度 72°:  phase = 1
角度 144°: phase = 2
角度 216°: phase = 3
角度 288°: phase = 4
角度 360°: phase = 0 (循环)
```

**运动曲线**:
```
Phase
  4 |      ╱      ╱
  3 |     ╱      ╱
  2 |    ╱      ╱
  1 |   ╱      ╱
  0 |__╱______╱___
    +--------------------> 角度
    0°   90°  180° 270° 360°
```

**特点**:
- ✅ 每个phase使用次数相同(均匀分布)
- ❌ 从phase 4到0有突变(不平滑)
- ⚠️ 适合测试或均匀采样,不太真实

---

## 3. Sawtooth (锯齿波)

**公式** (代码第96-97行):
```python
cycles = 2  # 2个周期
phase_continuous = (np.arange(num_angles) % (num_angles // cycles)) / (num_angles // cycles) * (num_phases - 1)
```

**工作原理**:
- 在指定周期数内线性增长
- 到达终点后立即跳回起点(锯齿状)
- 模拟快速呼气,缓慢吸气

**示例** (5个phase, 360角度, 2个周期):
```
角度 0°:   phase = 0
角度 45°:  phase = 1
角度 90°:  phase = 2
角度 135°: phase = 3
角度 180°: phase = 4 然后跳回...
角度 181°: phase = 0 (新周期开始)
```

**运动曲线**:
```
Phase
  4 |    ╱|    ╱|
  3 |   ╱ |   ╱ |
  2 |  ╱  |  ╱  |
  1 | ╱   | ╱   |
  0 |╱____|╱____|
    +--------------------> 角度
    0°   90°  180° 270° 360°
```

**特点**:
- ✅ 模拟不对称运动(快速返回)
- ✅ 可调节周期数
- ❌ 有明显的不连续点
- ⚠️ 适合特殊运动模式测试

---

## 投影混合过程 (代码第168-176行)

对于每个输出角度:

1. **确定phase**: `phase_idx = phase_indices[angle_idx]`
2. **计算源角度**: `source_angle_idx = int(angle_idx * num_proj_per_phase / num_angles)`
3. **加载投影**: 从对应phase的 `projection_{source_angle_idx:04d}.raw` 读取
4. **组合输出**: `mixed_projections[angle_idx] = projection`

**示例**: 输出角度90°
- Sinusoidal: phase=4 → 从phase_4文件夹读取第90个投影
- Linear: phase=1 → 从phase_1文件夹读取第90个投影
- Sawtooth: phase=2 → 从phase_2文件夹读取第90个投影

---

## 使用建议

| 场景 | 推荐方法 | 原因 |
|------|---------|------|
| **真实呼吸模拟** | Sinusoidal | 最接近生理运动 |
| **测试/验证** | Linear | 每个phase均匀采样 |
| **特殊运动** | Sawtooth | 不对称运动模式 |

---

## 可视化输出

运行程序后会生成 `motion_pattern_visualization.png`,包含:
- **上图**: Phase vs 角度曲线
- **下图**: 4个样例投影帧

这可以直观看到不同方法的运动模式差异。

---

## 实际应用

**正弦运动** 最常用,因为它:
1. 平滑过渡 → 减少重建伪影
2. 周期性 → 符合呼吸规律
3. Phase分布合理 → 呼气末/吸气末采样多,中间状态少

**参考文献**: 4DCT通常将呼吸周期分为10个phase,本程序支持任意数量的phase。