import torch
from torch import nn
import torch.nn.functional as F

class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):  # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


class GlobalAvgPool1d(nn.Module):
    # 全局平均池化层可通过将池化窗口形状设置成输入的高和宽实现
    def __init__(self):
        super(GlobalAvgPool1d, self).__init__()

    def forward(self, x):
        return F.avg_pool1d(x, kernel_size=x.size()[2:])


def conv_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm1d(input_channels), nn.ReLU(),
        nn.Conv1d(input_channels, num_channels, kernel_size=3, padding=1))


class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(
                num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # 连接通道维度上每个块的输入和输出
            X = torch.cat((X, Y), dim=1)
        return X


def transition_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm1d(input_channels), nn.ReLU(),
        nn.Conv1d(input_channels, num_channels, kernel_size=1),
        nn.AvgPool1d(kernel_size=2, stride=2))


b1 = nn.Sequential(
    nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm1d(64), nn.ReLU(),
    nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

num_channels, growth_rate = 64, 32
num_convs_in_dense_blocks = [4, 4, 4, 4]
blks = []
for i, num_convs in enumerate(num_convs_in_dense_blocks):
    blks.append(DenseBlock(num_convs, num_channels, growth_rate))
    num_channels += num_convs * growth_rate
    if i != len(num_convs_in_dense_blocks) - 1:
        blks.append(transition_block(num_channels, num_channels // 2))
        num_channels = num_channels // 2

DenseNet = nn.Sequential(
    b1, *blks,
    nn.BatchNorm1d(num_channels), nn.ReLU(),
    nn.AdaptiveAvgPool1d(1),
    nn.Flatten(),
    nn.Linear(num_channels, 474))
