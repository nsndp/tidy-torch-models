from functools import partial

import torch
import torch.nn as nn

from .basic import ConvUnit, BaseMultiReturn


# https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

class Bottleneck(nn.Module):

    def __init__(self, cin, width, stride, bn):
        super().__init__()
        w = width
        cout = w * 4
        self.u1 = ConvUnit(cin, w, 1, 1, 0, 'relu', bn)
        self.u2 = ConvUnit(w, w, 3, stride, 1, 'relu', bn)
        self.u3 = ConvUnit(w, cout, 1, 1, 0, 'relu', bn)
        if stride > 1 or cin != cout:
            self.downsample = ConvUnit(cin, cout, 1, stride, 0, activ=None, bn=bn)

    def forward(self, x):
        y = x if not hasattr(self, 'downsample') else self.downsample(x)
        x = self.u1(x)
        x = self.u2(x)
        x = self.u3(x, add=y)
        return x


class ResNet(BaseMultiReturn):

    def get_res_layer(self, cin, cout, stride, bn, count):
        blocks = [Bottleneck(cin, cout, stride, bn)]
        blocks.extend([Bottleneck(cout * 4, cout, 1, bn) for _ in range(1, count)])
        return nn.Sequential(*blocks)

    def __init__(self, block_counts, retidx=[1, 2, 3, 4], bn=1e-05, num_freeze=None):
        super().__init__(retidx)
        self.layers = nn.Sequential(
            nn.Sequential(
                ConvUnit(3, 64, k=7, s=2, p=3, activ='relu', bn=bn), # input_size/2
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)     # input_size/4
            ),
            self.get_res_layer(64, 64, 1, bn, block_counts[0]),   # C2 (input_size/4)
            self.get_res_layer(256, 128, 2, bn, block_counts[1]), # C3 (input_size/8)
            self.get_res_layer(512, 256, 2, bn, block_counts[2]), # C4 (input_size/16)
            self.get_res_layer(1024, 512, 2, bn, block_counts[3]) # C5 (imput_size/32)
        )
        if num_freeze:
            super().freeze(num_freeze)


ResNet50 = partial(ResNet, block_counts=[3, 4, 6, 3])
ResNet152 = partial(ResNet, block_counts=[3, 8, 36, 3])