# 对于transformer中的embeding过程，如何对token和pos进行编码
# 以一个224*224*3大小的图像为例
import torch
from torch import nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def read_jpg_to_numpy(image_path):
    """
    读取.jpg图像并转换为numpy数组
    参数:image_path: str, 图像文件路径
    返回:numpy.ndarray: 图像的numpy数组表示
    """
    # 方法1: 使用PIL/Pillow读取
    img = Image.open(image_path)
    img_array = np.array(img, dtype=np.float32)
    
    return img_array

# 最常用的设计，使用卷积进行patch embedding
# 能够极大地节省计算资源
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # 输入: (B, 3, 224, 224), B为Batch大小
        x = self.proj(x)  # 输出: (B, 768, 14, 14)
        x = x.flatten(2).transpose(1, 2)  # (B, 196, 768)
        return x


if __name__ == "__main__":
    # 读取图像为numpy数组
    image_path = '2021051521244130.jpg'
    img_array = read_jpg_to_numpy(image_path)
    print(f"读取图像: {image_path}")
    print(f"图像形状: {img_array.shape}")
    print(f"数据类型: {img_array.dtype}")

    # 将图像进行根据patch大小embeding
    # 一般使用CNN网络进行embeding
    patch_size = 16
    num_patches = (224 // patch_size) * (224 // patch_size)
    embedding_dim = 768  # 一般选择较大的维度
    vocab_size = num_patches  # 这里将每个patch视为一个token
    embedding_layer = PatchEmbedding(img_size=224, patch_size=patch_size,
                                     in_chans=3, embed_dim=embedding_dim)
    print(f"patch数量: {num_patches}, embedding维度: {embedding_dim}")
    print(f"对patch进行embeding...")
    embeded = embedding_layer.forward(torch.tensor(img_array).unsqueeze(0).permute(0, 3, 1, 2))
    print(f"Embeding后形状: {embeded.shape}")  # (1, 196, 768)
    
    # 对位置pos进行编码
    ##生成768维的embedding
    output_dim = embedding_dim
    ##使用制定的种子，确保可以复现
    torch.manual_seed(1111)
    embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    print(embedding_layer.weight.shape)  # (196, 768)
    #取第2个Token的embedding，python是从0开始。前面每一个TokenID都可以作为下标从而Embedings矩阵中获取。
    # print(embedding_layer(torch.tensor([1]))) # 显示了某一个位置编码

    # 将位置编码embeding和patch embedding相加就可以得到最终的输入表示
    final_embedding = embeded + embedding_layer.weight.unsqueeze(0)
    print(f"最终的输入表示形状: {final_embedding.shape}")  # (1, 196, 768)
    # 1个batch中有196个token，每个token是768维的向量表示