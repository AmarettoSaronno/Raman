from torch import nn
import torch.nn.functional as F

class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):  # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


class GlobalAvgPool1d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool1d, self).__init__()

    def forward(self, x):
        return F.avg_pool1d(x, kernel_size=x.size()[2:])


def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv1d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


conv_arch = ((2, 64), (2, 128), (4, 256), (4, 512), (4, 512))


def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        nn.Linear(23552, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 2048), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(2048, 474))