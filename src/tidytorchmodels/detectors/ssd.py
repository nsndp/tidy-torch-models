from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbones.basic import ConvUnit, BaseMultiReturn
from ..backbones.mobilenet import MobileNetV2, MobileNetV3L
from .operations.anchor import get_priors, make_anchors
from .operations.bbox import clamp_to_canvas, decode_boxes, remove_small, scale_boxes
from .operations.post import top_per_class, top_per_level, final_nms
from .operations.prep import preprocess
from ..utils.weights import load_weights

# SSD paper: https://arxiv.org/pdf/1512.02325.pdf
# VGG paper: https://arxiv.org/pdf/1409.1556.pdf


def vgg_convunit(cin, cout, k, s, p, d=1):
    return ConvUnit(cin, cout, k, s, p, 'relu', bn=None, d=d)


def mbnet_convunit(cin, cout, k, s, p, grp=1):
    return ConvUnit(cin, cout, k, s, p, 'relu6', 1e-03, grp=grp)


class VGG16(BaseMultiReturn):
    
    def __init__(self):
        super().__init__()
        # configuration D from the paper
        cfg = [(64, 2), (128, 2), (256, 3), (512, 3), (512, 3)]
        layers = []
        cin = 3
        for c, n in cfg:
            for i in range(n):
                layers.append(vgg_convunit(cin, c, 3, 1, 1))
                cin = c
            layers.append(nn.MaxPool2d(2))
        self.layers = nn.Sequential(*layers)
    
    
class ExtendedVGG16(nn.Module):

    def __init__(self, hires=False):
        super().__init__()
        self.l2_scale = nn.Parameter(torch.ones(512) * 20)
        self.backbone = VGG16()
        self.backbone.layers[9].ceil_mode = True   # adjusting MaxPool3
        # the above is needed in SSD_300 to go from 75 pixels to 38 (and not 37)
        # for other pools (and for all SSD_512), the input size will be even so it doesn't matter
        self.backbone.layers.pop(-1)               # removing MaxPool5
        self.backbone.layers.extend([
            nn.MaxPool2d(3, 1, 1),                 # replacement for MaxPool5
            vgg_convunit(512, 1024, 3, 1, 6, d=6), # FC6
            vgg_convunit(1024, 1024, 1, 1, 0)      # FC7
        ])
        self.backbone.retidx = [12, 19] # right before MaxPool4, last layer with extensions

        self.extra = nn.ModuleList()
        settings = [(1024, 512, 3, 2, 1), (512, 256, 3, 2, 1)]
        if not hires:
            settings.extend([(256, 256, 3, 1, 0), (256, 256, 3, 1, 0)])
        else:
            settings.extend([(256, 256, 3, 2, 1), (256, 256, 3, 2, 1), (256, 256, 4, 1, 1)])
        for cin, cout, k, s, p in settings:
            self.extra.append(nn.Sequential(
                vgg_convunit(cin, cout // 2, 1, 1, 0),
                vgg_convunit(cout // 2, cout, k, s, p)
            ))
        self.couts = [512, 1024] + [s[1] for s in settings]

    def forward(self, x):
        xs = self.backbone(x)
        if hasattr(self, 'l2_scale'):
            xs[0] = F.normalize(xs[0]) * self.l2_scale.view(1, -1, 1, 1)
        x = xs[1]
        for block in self.extra:
            x = block(x)
            xs.append(x)
        return xs


class ExtendedMobileNet(nn.Module):

    def __init__(self, version):
        super().__init__()
        assert version in ['mobile2', 'mobile3']
        if version == 'mobile2':
            self.backbone = MobileNetV2([5, 8], 'relu6', 1e-03)
            bbone_couts = [96, 1280]
        else:
            self.backbone = MobileNetV3L([13, 17], bn=1e-03, reduce_tail=True)
            lr = self.backbone.layers[13]
            self.backbone.layers.insert(14, lr.block[1:])
            self.backbone.layers[13] = lr.block[0]
            bbone_couts = [672, 480]
        self.extra = nn.ModuleList()
        settings = [(bbone_couts[-1], 512), (512, 256), (256, 256), (256, 128)]
        for cin, cout in settings:
            cmid = cout // 2
            self.extra.append(nn.Sequential(
                mbnet_convunit(cin, cmid, 1, 1, 0),
                nn.Sequential(
                    mbnet_convunit(cmid, cmid, 3, 2, 1, grp=cmid),
                    mbnet_convunit(cmid, cout, 1, 1, 0)
                )
            ))
        self.couts = bbone_couts + [s[1] for s in settings]

    def forward(self, x):
        xs = self.backbone(x)
        x = xs[1]
        for block in self.extra:
            x = block(x)
            xs.append(x)
        return xs


class SSDHead(nn.Module):

    def __init__(self, bbone, c, num_anchors, task_len):
        super().__init__()
        self.task_len = task_len
        if bbone != 'vgg16':
            self.conv = nn.Sequential(mbnet_convunit(c, c, 3, 1, 1, grp=c),
                                      nn.Conv2d(c, num_anchors * task_len, 1, 1, 0))
        else:
            self.conv = nn.Conv2d(c, num_anchors * task_len, 3, 1, 1)
    
    def forward(self, x):
        x = self.conv(x).permute(0, 2, 3, 1)
        x = x.reshape(x.shape[0], -1, self.task_len)
        return x


class SSD(nn.Module):

    def __init__(self, device='cpu', backbone='vgg16', canvas_size=300, num_classes=80, bckg_first=False,
                 anchors_scales=None, anchors_clamp=False, resize='cv2', means='imagenet', stdvs=None,
                 lvtop=1000, cltop=None, imtop=200, score_thr=0.02, nms_thr=0.45):
        super().__init__()
        assert backbone in ['vgg16', 'mobile2', 'mobile3']
        self.canvas_size = canvas_size
        self.bckg_first = bckg_first
        self.backend = resize
        self.meanstd = (means, stdvs)
        self.settings = (lvtop, cltop, imtop, score_thr, nms_thr)
        
        if backbone == 'vgg16':
            self.backbone = ExtendedVGG16(canvas_size == 512)
        elif backbone.startswith('mobile'):
            self.backbone = ExtendedMobileNet(backbone)
        self.bases, num_anchors_per_level = self.get_bases(backbone, canvas_size, anchors_clamp)
        level_dims = list(zip(self.backbone.couts, num_anchors_per_level))
        self.cls_heads = nn.ModuleList([SSDHead(backbone, c, an, num_classes + 1) for c, an in level_dims])
        self.reg_heads = nn.ModuleList([SSDHead(backbone, c, an, 4) for c, an in level_dims])
    
    def forward(self, imgs):
        dv = next(self.parameters()).device
        x, sz_orig, sz_used = preprocess(imgs, dv, self.canvas_size, self.backend, False, 1, *self.meanstd)
        xs = self.backbone(x)
        cls = [self.cls_heads[i](xs[i]) for i in range(len(xs))]
        reg = [self.reg_heads[i](xs[i]) for i in range(len(xs))]
        lvlen = [lvl.shape[1] for lvl in cls]
        cls = torch.cat(cls, dim=1)
        reg = torch.cat(reg, dim=1)
        scr = F.softmax(cls, dim=-1)
        scr = scr[:, :, :-1] if not self.bckg_first else scr[:, :, 1:]
        priors = get_priors(x.shape[2:], self.bases)
        b, s, c = self.postprocess(reg, scr, priors, sz_used, lvlen)
        b = scale_boxes(b, sz_orig, sz_used)
        b, s, c = [[t.detach().cpu().numpy() for t in tl] for tl in [b, s, c]]
        return b, s, c

    strides_and_ar3flags = {
        '300': ([8, 16, 32, 64, 100, 300], [False, True, True, True, False, False]),
        '512': ([8, 16, 32, 64, 128, 256, 512], [False, True, True, True, True, False, False]),
        '320': ([16, 32, 64, 106.67, 160, 320], [True] * 6)
    }
    
    def get_scales(self, backbone, canvas_size):
        # special cases
        if backbone == 'mobile2':
            scales = np.append(np.linspace(0.15, 0.95, 6), 1)
            scales[1] += 0.0025 # manual adjustment to match MMDet boxes
            return scales
        if backbone == 'mobile3':
            return np.append(np.linspace(0.2, 0.95, 6), 1) # Eq.4 SSD paper
        # vgg16 route            
        # formula from official implementation (which is like eq.4 from SSD paper but not exactly):
        # https://github.com/weiliu89/caffe/blob/ssd/examples/ssd/ssd_coco.py#L315
        # + section 3.4 of SSD paper (smallest boxes for COCO)
        smallest, rmin, rmax, n = (0.07, 0.15, 0.87, 6) if canvas_size == 300 else (0.04, 0.1, 0.9, 7)
        step = (rmax - rmin) / (n - 2)
        scales = [smallest] + [rmin + i * step for i in range(n)]
        return np.array(scales)

    def get_bases(self, backbone, canvas_size, anchors_clamp=False):
        # custom SDD anchors as described on page 6 of the paper
        assert canvas_size in [300, 512, 320]
        strides, ar3flags = self.strides_and_ar3flags[str(canvas_size)]
        scales = self.get_scales(backbone, canvas_size)
        sz = (scales * canvas_size).round().astype(int)
        anchors = []
        for i in range(len(strides)):
            r = [1, 2, 0.5] + ([3, 1/3] if ar3flags[i] else [])
            a = make_anchors([sz[i]], ratios=r)[0]
            extra = np.sqrt(sz[i + 1] / sz[i]) * sz[i]
            a.insert(1, (extra, extra))
            if anchors_clamp:
                a = [(min(x, canvas_size), min(y, canvas_size)) for x, y in a]
            anchors.append(a)
        return list(zip(strides, anchors)), [len(a) for a in anchors]

    def postprocess(self, reg, scr, priors, sz_used, lvlen):
        lvtop, cltop, imtop, score_thr, nms_thr = self.settings
        n, dim, num_classes = scr.shape
        reg, scr = reg.reshape(-1, 4), scr.flatten()
        fidx = torch.nonzero(scr > score_thr).squeeze()
        fidx = top_per_level(fidx, scr, lvtop, lvlen, n, mult=num_classes)
        scores = scr[fidx]
        classes = fidx % num_classes + (0 if not self.bckg_first else 1)
        idx = torch.div(fidx, num_classes, rounding_mode='floor')
        imidx = idx.div(dim, rounding_mode='floor')

        if cltop:
            sel = top_per_class(scores, classes, imidx, cltop)
            scores, classes, imidx, idx = [x[sel] for x in [scores, classes, imidx, idx]]
        
        boxes = decode_boxes(reg[idx], priors[idx % dim], mults=(0.1, 0.2), clamp=True)
        boxes = clamp_to_canvas(boxes, sz_used, imidx)
        boxes, scores, classes, imidx = remove_small(boxes, 0, scores, classes, imidx)
        return final_nms(boxes, scores, classes, imidx, n, nms_thr, imtop)


class Detector_SSD():

    tvhub = 'https://download.pytorch.org/models/'
    mmhub = 'https://download.openmmlab.com/mmdetection/v2.0/ssd/'
    links = {
        'vgg16_300_tv': tvhub + 'ssd300_vgg16_coco-b556d3b4.pth',
        'vgg16_300_mm': mmhub + 'ssd300_coco/ssd300_coco_20210803_015428-d231a06e.pth',
        'vgg16_512_mm': mmhub + 'ssd512_coco/ssd512_coco_20210803_022849-0a47a1ca.pth',
        'mobile2_320_mm': mmhub + 'ssdlite_mobilenetv2_scratch_600e_coco/ssdlite_mobilenetv2'\
                          '_scratch_600e_coco_20210629_110627-974d9307.pth',
        'mobile3_320_tv': tvhub + 'ssdlite320_mobilenet_v3_large_coco-a79551df.pth'
    }

    def mm_vgg_conversion(self, wd):
        nm = 'neck.l2_norm.weight'
        ret = {nm: wd.pop(nm)}
        ret.update(wd)
        return ret

    def __init__(self, src, device=None, train=False):
        assert src in self.links
        print('Initializing %s model for object detection' % 'SSD')
        dv = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')

        SSDTV = partial(SSD, num_classes=90, bckg_first=True, anchors_clamp=True,
                        resize='torch', lvtop=None, score_thr=0.01)

        if src == 'vgg16_300_mm':
            self.model = SSD(dv)
        elif src == 'vgg16_512_mm':
            self.model = SSD(dv, canvas_size=512)
        elif src == 'vgg16_300_tv':
            self.model = SSDTV(dv, cltop=400, means=(122.99925, 116.9991, 103.9992)) #~(0.48235, 0.45882, 0.40784)
        elif src == 'mobile2_320_mm':
            self.model = SSD(dv, 'mobile2', 320, means='imagenet', stdvs='imagenet')
        elif src == 'mobile3_320_tv':
            self.model = SSDTV(dv, 'mobile3', 320, means=127.5, stdvs=127.5,
                               cltop=300, imtop=300, score_thr=0.001, nms_thr=0.55)
        
        wconv = None if (src != 'vgg16_300_mm' and src != 'vgg16_512_mm') else self.mm_vgg_conversion
        wsub = 'state_dict' if src.endswith('mm') else None
        load_weights(self.model, self.links[src], 'ssd_' + src, wconv, wsub)

        if not train:
            self.model.eval()

    def __call__(self, imgs, targets=None, seed=None):
        if self.model.training:
            if seed is not None:
                torch.manual_seed(seed)
            return self.model(imgs, targets)
        else:
            with torch.inference_mode():
                return self.model(imgs)        