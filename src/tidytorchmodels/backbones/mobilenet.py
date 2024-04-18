import torch
import torch.nn as nn

from .basic import ConvUnit, BaseMultiReturn

# V1: https://arxiv.org/abs/1704.04861
# V2: https://arxiv.org/pdf/1801.04381
# V3: https://arxiv.org/pdf/1905.02244
# https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/backbones/mobilenet_v2.py
# https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py
  

class MobileNetV1(BaseMultiReturn):

    def depthwise_block(self, cin, cout, s):
        return nn.Sequential(
            ConvUnit(cin, cin, 3, s, 1, self.activ, self.bn, grp=cin),
            ConvUnit(cin, cout, 1, 1, 0, self.activ, self.bn)
        )

    def __init__(self, width_multiplier, activ, bn=1e-05, retidx=None):
        super().__init__(retidx)
        a = width_multiplier
        self.activ = activ
        self.bn = bn
        self.layers = nn.Sequential(
            ConvUnit(3, 32*a, 3, 2, 1, activ, bn),
            self.depthwise_block(32*a, 64*a, 1),
            self.depthwise_block(64*a, 128*a, 2),
            self.depthwise_block(128*a, 128*a, 1),
            self.depthwise_block(128*a, 256*a, 2),
            self.depthwise_block(256*a, 256*a, 1),
            self.depthwise_block(256*a, 512*a, 2),
            *[self.depthwise_block(512*a, 512*a, 1) for _ in range(5)],
            self.depthwise_block(512*a, 1024*a, 2),
            self.depthwise_block(1024*a, 1024*a, 1)
        )


class SqueezeExcitation(nn.Module):

    def __init__(self, cin, csq):
        super().__init__()
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = torch.nn.Conv2d(cin, csq, 1)
        self.fc2 = torch.nn.Conv2d(csq, cin, 1)
        self.activation = nn.ReLU()
        self.scale_activation = nn.Hardsigmoid()

    def forward(self, x):
        s = self.avgpool(x)
        s = self.fc1(s)
        s = self.activation(s)
        s = self.fc2(s)
        s = self.scale_activation(s)
        return x * s


def make_divisable(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class InvertedResidual(nn.Module):

    def __init__(self, cin, k, cmid, cout, use_se, activ, bn, stride):
        super().__init__()
        layers = []
        activ = activ.replace('RE', 'relu').replace('HS', 'hardswish')
        if cin != cmid:
            layers.append(ConvUnit(cin, cmid, 1, 1, 0, activ, bn))
        layers.append(ConvUnit(cmid, cmid, k, stride, (k - 1) // 2, activ, bn, grp=cmid))
        if use_se:
            csq = make_divisable(cmid // 4, divisor=8)
            layers.append(SqueezeExcitation(cmid, csq))
        layers.append(ConvUnit(cmid, cout, 1, 1, 0, None, bn))
        
        self.block = nn.Sequential(*layers)
        self.residual = stride == 1 and cin == cout

    def forward(self, x):
        y = self.block(x)
        if self.residual:
            y += x
        return y


class MobileNetV2(BaseMultiReturn):

    def get_layer(self, t, c, n, s, cin):
        return nn.Sequential(
            InvertedResidual(cin, 3, cin*t, c, False, self.activ, self.bn, s),
            *[InvertedResidual(c, 3, c*t, c, False, self.activ, self.bn, 1) for i in range(1, n)]
        )
    
    def __init__(self, retidx=None, activ='lrelu_0.1', bn=1e-05):
        super().__init__(retidx)
        self.activ = activ
        self.bn = bn
        self.layers = nn.Sequential(
            ConvUnit(3, 32, 3, 2, 1, activ, bn),
            self.get_layer(1, 16, 1, 1, cin=32),
            self.get_layer(6, 24, 2, 2, cin=16),
            self.get_layer(6, 32, 3, 2, cin=24),
            self.get_layer(6, 64, 4, 2, cin=32),
            self.get_layer(6, 96, 3, 1, cin=64),
            self.get_layer(6, 160, 3, 2, cin=96),
            self.get_layer(6, 320, 1, 1, cin=160),
            ConvUnit(320, 1280, 1, 1, 0, activ, bn)
        )


class MobileNetV3L(BaseMultiReturn):

    def __init__(self, retidx=None, bn=1e-05, reduce_tail=False, num_freeze=None):
        super().__init__(retidx)
        out = 160
        if reduce_tail:
            out = out // 2
        self.layers = nn.Sequential(
            ConvUnit(3, 16, 3, 2, 1, 'hardswish', bn),
            InvertedResidual( 16, 3,  16,  16, False, 'RE', bn, 1),
            InvertedResidual( 16, 3,  64,  24, False, 'RE', bn, 2), # C1
            InvertedResidual( 24, 3,  72,  24, False, 'RE', bn, 1),
            InvertedResidual( 24, 5,  72,  40, True,  'RE', bn, 2), # C2
            InvertedResidual( 40, 5, 120,  40, True,  'RE', bn, 1),
            InvertedResidual( 40, 5, 120,  40, True,  'RE', bn, 1),
            InvertedResidual( 40, 3, 240,  80, False, 'HS', bn, 2), # C3
            InvertedResidual( 80, 3, 200,  80, False, 'HS', bn, 1),
            InvertedResidual( 80, 3, 184,  80, False, 'HS', bn, 1),
            InvertedResidual( 80, 3, 184,  80, False, 'HS', bn, 1),
            InvertedResidual( 80, 3, 480, 112, True,  'HS', bn, 1),
            InvertedResidual(112, 3, 672, 112, True,  'HS', bn, 1),
            InvertedResidual(112, 5, 672, out, True,  'HS', bn, 2), # C4
            InvertedResidual(out, 5,6*out,out, True,  'HS', bn, 1),
            InvertedResidual(out, 5,6*out,out, True,  'HS', bn, 1),
            ConvUnit(out, 6*out, 1, 1, 0, 'hardswish', bn)
        )
        if num_freeze:
            super().freeze(num_freeze)