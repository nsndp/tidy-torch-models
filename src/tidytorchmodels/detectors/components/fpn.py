import torch
import torch.nn as nn
import torch.nn.functional as F

from ...backbones.basic import ConvUnit


class FeaturePyramidNetwork(nn.Module):
    """
    FPN paper (section 3 and figure 3) https://arxiv.org/pdf/1612.03144.pdf
    RetinaNet paper (page 4 footnote 2) https://arxiv.org/pdf/1708.02002.pdf
    RetinaFace paper (start of section 4.2) https://arxiv.org/pdf/1905.00641.pdf

    example: https://github.com/kuangliu/pytorch-fpn/blob/master/fpn.py
    tvision: https://github.com/pytorch/vision/blob/main/torchvision/ops/feature_pyramid_network.py

    ResNet outputs: C2, C3, C4, C5 ({4, 8, 16, 32} stride w.r.t. the input image)
        
    Ti = lateral(Ci) [i=2..5]
    P5 = T5
    P4 = T4 + upsample(P5)
    P3 = T3 + upsample(P4)
    P2 = T2 + upsample(P2)
    Pi = smooth(Pi) [i=2..4] (or 5 too)

    P6 = extra1(C5) [or P5]
    P7 = extra2(relu(P6))

    smoothP5 is probably not canon, but both torchvision and detectron2 implementations seem to be using it
    from the paper, same paragraph: "which is to reduce the aliasing effect of upsampling" (but P5 have no upsampling)
    """

    def __init__(self, cins, cout, activ, bn=1e-05, P6=None, P7=None, pool=False,
                 smoothP5=False, smoothBeforeMerge=False, nonCumulative=False):
        super().__init__()
        assert P6 in ['fromC5', 'fromP5', None]
        if pool:
            assert (not P6) and (not P7)
        self.P6 = P6
        self.P7 = P7
        self.pool = pool
        self.smoothBeforeMerge = smoothBeforeMerge
        self.nonCumulative = nonCumulative
        smooth_n = len(cins) - (0 if smoothP5 else 1)
        self.conv_laterals = nn.ModuleList([ConvUnit(cin, cout, 1, 1, 0, activ, bn) for cin in cins])
        self.conv_smooths = nn.ModuleList([ConvUnit(cout, cout, 3, 1, 1, activ, bn) for _ in range(smooth_n)])
        if P6:
            cin6 = cins[-1] if P6 == 'fromC5' else cout
            self.conv_extra1 = ConvUnit(cin6, cout, 3, 2, 1, activ, bn)
        if P7:
            self.conv_extra2 = ConvUnit(cout, cout, 3, 2, 1, activ, bn)

    def forward(self, C):
        n = len(C)
        P = [self.conv_laterals[i](C[i]) for i in range(n)]
        
        if self.nonCumulative:
            # peculiarity from: https://github.com/barisbatuhan/FaceDetector/blob/main/BBTNet/components/fpn.py
            P = [P[i] + F.interpolate(P[i + 1], size=P[i].shape[2:], mode='nearest') for i in range(len(P) - 1)] + [P[-1]]
            for i in range(len(self.conv_smooths)):
                P[i] = self.conv_smooths[i](P[i])
        elif self.smoothBeforeMerge:
            # peculiarity from: https://github.com/biubug6/Pytorch_Retinaface/blob/master/models/net.py
            # from the paper: "Finally, we append a 3×3 convolution on each merged map"
            # P5 is never smoothed here (smoothP5 is ignored)
            for i in range(n - 1)[::-1]:
                P[i] += F.interpolate(P[i + 1], size=P[i].shape[2:], mode='nearest')
                P[i] = self.conv_smooths[i](P[i])
        else:
            # normal pathway
            for i in range(n - 1)[::-1]:
                P[i] += F.interpolate(P[i + 1], size=P[i].shape[2:], mode='nearest')
            for i in range(len(self.conv_smooths)):
                P[i] = self.conv_smooths[i](P[i])
        
        if self.P6:
            P.append(self.conv_extra1(C[-1] if self.P6 == 'fromC5' else P[-1]))
        if self.P7:
            P.append(self.conv_extra2(F.relu(P[-1])))
        if self.pool:
            P.append(F.max_pool2d(P[-1], 1, stride=2))

        return P