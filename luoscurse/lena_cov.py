"""
实现 Lena 原始图像的卷积操作，并使用 matplotlib 以窗口形式展示多种卷积核的结果。

Usage 示例:
    python lena_cov.py --kernels blur3 sharpen laplacian
    python lena_cov.py --kernels custom_kernel.txt sobel_x

运行脚本前请确保安装 matplotlib (pip install matplotlib)。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "需要安装 matplotlib 库以可视化卷积后的图像，请执行: pip install matplotlib"
    ) from exc

# Lena.raw 的默认尺寸与类型
IMAGE_SIZE: Tuple[int, int] = (512, 512)
IMAGE_DTYPE = np.uint8

# 预置的卷积核，可通过 --kernels 参数直接引用名称
DEFAULT_KERNELS: Dict[str, np.ndarray] = {
    "blur3": np.ones((3, 3), dtype=np.float32) / 9.0,
    "sharpen": np.array(
        [
            [0.0, -1.0, 0.0],
            [-1.0, 5.0, -1.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    ),
    "laplacian": np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, -4.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    ),
    "sobel_x": np.array(
        [
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    ),
    "sobel_y": np.array(
        [
            [-1.0, -2.0, -1.0],
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 1.0],
        ],
        dtype=np.float32,
    ),
}


def load_lena_image(image_path: Path, size: Tuple[int, int]) -> np.ndarray:
    """从原始 raw 文件读取 Lena 图像。"""
    expected_elements = size[0] * size[1]
    data = np.fromfile(image_path, dtype=IMAGE_DTYPE)
    if data.size != expected_elements:
        raise ValueError(
            f"图像尺寸不匹配: 期望 {expected_elements} 个像素, 实际 {data.size}."
        )
    return data.reshape(size)


def load_kernel_from_path(path: Path) -> np.ndarray:
    """从文本文件构建卷积核，文件中每行使用空格或逗号分隔。"""
    if not path.exists():
        raise FileNotFoundError(f"未找到卷积核文件: {path}")
    try:
        kernel = np.loadtxt(path, dtype=np.float32, delimiter=None)
    except ValueError as exc:
        raise ValueError(f"解析卷积核文件失败: {path}") from exc

    if kernel.ndim != 2:
        raise ValueError(f"卷积核必须是二维矩阵: {path}")
    return kernel


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """手写二维卷积 (same padding)。"""
    if image.ndim != 2:
        raise ValueError("仅支持二维灰度图像的卷积。")

    kernel = np.flipud(np.fliplr(np.asarray(kernel, dtype=np.float32)))
    kh, kw = kernel.shape
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError("卷积核尺寸必须为奇数。")

    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image.astype(np.float32), ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")
    convolved = np.zeros_like(image, dtype=np.float32)

    for row in range(convolved.shape[0]):
        for col in range(convolved.shape[1]):
            region = padded[row : row + kh, col : col + kw]
            convolved[row, col] = np.sum(region * kernel)
    return convolved


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """将卷积结果线性归一化到 0~255。"""
    arr_min = arr.min()
    arr_max = arr.max()
    if np.isclose(arr_max, arr_min):
        return np.zeros_like(arr, dtype=np.uint8)
    normalized = (arr - arr_min) / (arr_max - arr_min)
    return (normalized * 255.0).clip(0, 255).astype(np.uint8)


def parse_kernel_specs(specs: Iterable[str]) -> Iterable[Tuple[str, np.ndarray]]:
    """解析用户输入的卷积核规格 (名称或文件路径)。"""
    for spec in specs:
        key = spec.lower()
        if key in DEFAULT_KERNELS:
            # 返回副本避免默认卷积核被原地修改
            yield key, DEFAULT_KERNELS[key].copy()
            continue

        kernel_path = Path(spec)
        kernel = load_kernel_from_path(kernel_path)
        yield kernel_path.stem, kernel


def display_results(results: List[Tuple[str, np.ndarray]]) -> None:
    """使用 matplotlib 展示卷积后的图像结果。"""
    if not results:
        print("未生成任何卷积结果。")
        return

    count = len(results)
    cols = min(3, count)
    rows = int(np.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes_arr = np.atleast_1d(axes).reshape(rows, cols)

    for ax, (kernel_name, image) in zip(axes_arr.flat, results):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(kernel_name)
        ax.axis("off")

    for ax in axes_arr.flat[len(results) :]:
        ax.axis("off")

    fig.suptitle("Lena Convolution Results")
    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="对 Lena.raw 执行二维卷积，可选择预置或自定义卷积核，并通过窗口展示结果。",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("Lena.raw"),
        help="输入的 raw 图像路径 (默认: 当前目录下的 Lena.raw)。",
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=list(IMAGE_SIZE),
        metavar=("HEIGHT", "WIDTH"),
        help="图像尺寸，默认 512 512。",
    )
    parser.add_argument(
        "--kernels",
        "-k",
        nargs="+",
        help="卷积核名称或文件路径，可指定多个；默认使用 blur3、sharpen、laplacian。",
    )

    args = parser.parse_args()

    image = load_lena_image(args.image, tuple(args.size))

    kernel_specs = args.kernels or ["blur3", "sharpen", "laplacian"]
    if len(kernel_specs) < 3:
        raise ValueError("请至少指定三个卷积核，以满足需求。")

    results: List[Tuple[str, np.ndarray]] = []
    for kernel_name, kernel in parse_kernel_specs(kernel_specs):
        result = convolve2d(image, kernel)
        normalized = normalize_to_uint8(result)
        print(f"卷积完成: {kernel_name}")
        results.append((kernel_name, normalized))

    display_results(results)


if __name__ == "__main__":
    main()
