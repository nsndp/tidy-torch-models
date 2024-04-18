import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbones.basic import ConvUnit
from ..backbones.mobilenet import MobileNetV1
from ..backbones.resnet import ResNet50, ResNet152
from .components.fpn import FeaturePyramidNetwork
from .operations.anchor import get_priors, make_anchors
from .operations.bbox import clamp_to_canvas, decode_boxes, scale_boxes
from .operations.loss import get_losses
from .operations.post import get_results, top_per_level, final_nms
from .operations.prep import prep_targets, preprocess, to_tensors
from ..utils.weights import load_weights

# Source 1: https://github.com/biubug6/Pytorch_Retinaface
# Source 2: https://github.com/barisbatuhan/FaceDetector
# Paper: https://arxiv.org/pdf/1905.00641.pdf


class SSH(nn.Module):

    def __init__(self, cin, cout, activ):
        super().__init__()
        self.conv1 = ConvUnit(cin, cout//2, 3, 1, 1, activ=None)
        self.conv2 = ConvUnit(cin, cout//4, 3, 1, 1, activ=activ)
        self.conv3 = ConvUnit(cout//4, cout//4, 3, 1, 1, activ=None)
        self.conv4 = ConvUnit(cout//4, cout//4, 3, 1, 1, activ=activ)
        self.conv5 = ConvUnit(cout//4, cout//4, 3, 1, 1, activ=None)

    def forward(self, x):
        y1 = self.conv1(x)
        t = self.conv2(x)
        y2 = self.conv3(t)
        y3 = self.conv5(self.conv4(t))
        out = torch.cat([y1, y2, y3], dim=1)
        out = F.relu(out)
        return out


class Head(nn.Module):

    def __init__(self, cin, num_anchors, task_len):
        super().__init__()
        self.task_len = task_len
        self.conv = nn.Conv2d(cin, num_anchors * task_len, kernel_size=1)
    
    def forward(self, x):
        x = self.conv(x).permute(0, 2, 3, 1)
        x = x.reshape(x.shape[0], -1, self.task_len)
        return x


class HeadShared(nn.Module):

    def __init__(self, c, num_anchors, task_len):
        super().__init__()
        self.task_len = task_len
        self.conv = nn.Sequential(*[ConvUnit(c, c, 3, 1, 1, activ='relu', bn=None) for _ in range(4)])
        self.final = nn.Conv2d(c, num_anchors * task_len, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.final(self.conv(x)).permute(0, 2, 3, 1)
        x = x.reshape(x.shape[0], -1, self.task_len)
        return x
    

class RetinaFace_Biubug6(nn.Module):

    def __init__(self, device='cpu', mobile=True):
        super().__init__()
        if mobile:
            backbone = MobileNetV1(0.25, activ='lrelu_0.1', retidx=[5, 11, 13])
            cins, cout, activ = [64, 128, 256], 64, 'lrelu_0.1'
        else:
            backbone = ResNet50(retidx=[2, 3, 4])
            cins, cout, activ = [512, 1024, 2048], 256, 'relu'
        self.bases = list(zip([8, 16, 32], make_anchors([16, 64, 256], scales=[1, 2])))
        num_anchors = 2
        self.body = backbone
        self.feature_pyramid = FeaturePyramidNetwork(cins, cout, activ, smoothBeforeMerge=True)
        self.context_modules = nn.ModuleList([SSH(cout, cout, activ) for _ in range(len(cins))])
        self.heads_class = nn.ModuleList([Head(cout, num_anchors, 2) for _ in range(len(cins))])
        self.heads_boxes = nn.ModuleList([Head(cout, num_anchors, 4) for _ in range(len(cins))])
        self.heads_ldmks = nn.ModuleList([Head(cout, num_anchors, 10) for _ in range(len(cins))])
        self.to(device)
        
    def forward(self, imgs):
        dv = next(self.parameters()).device
        ts = to_tensors(imgs, dv, means=[104, 117, 123], stdvs=None, to_rgb=False)
        x = torch.stack(ts)

        xs = self.body(x)
        xs = self.feature_pyramid(xs)
        xs = [self.context_modules[i](xs[i]) for i in range(len(xs))]
        reg = torch.cat([self.heads_boxes[i](xs[i]) for i in range(len(xs))], dim=1)
        cls = torch.cat([self.heads_class[i](xs[i]) for i in range(len(xs))], dim=1)
        #ldm = torch.cat([self.heads_ldmks[i](xs[i]) for i in range(len(xs))], dim=1)
        
        scr = F.softmax(cls, dim=-1)[:, :, 1]
        priors = get_priors(x.shape[2:], self.bases, dv, loc='center')
        b, s = get_results(reg, scr, priors, 0.02, 0.4, decode_mults=(0.1, 0.2))
        b, s = [[t.detach().cpu().numpy() for t in tl] for tl in [b, s]]
        return b, s


class RetinaFace_BBT(nn.Module):

    def __init__(self, device='cpu', resnet50=True):
        super().__init__()
        backbone = ResNet50() if resnet50 else ResNet152()
        cins = [256, 512, 1024, 2048]
        cout = 256
        activ='relu'
        anchors = make_anchors([16, 32, 64, 128, 256], scales=[1, 2**(1/3), 2**(2/3)])
        self.bases = list(zip([4, 8, 16, 32, 64], anchors))
        num_anchors = 3
        self.body = backbone
        self.feature_pyramid = FeaturePyramidNetwork(cins, cout, activ, P6='fromC5', nonCumulative=True)
        self.context_modules = nn.ModuleList([SSH(cout, cout, activ) for _ in range(len(cins) + 1)])
        self.heads_class = nn.ModuleList([Head(cout, num_anchors, 2) for _ in range(len(cins) + 1)])
        self.heads_boxes = nn.ModuleList([Head(cout, num_anchors, 4) for _ in range(len(cins) + 1)])
        self.to(device)
    
    def forward(self, imgs):
        dv = next(self.parameters()).device
        ts = to_tensors(imgs, dv, means='imagenet', stdvs='imagenet')
        x = torch.stack(ts)
        
        xs = self.body(x)
        xs = self.feature_pyramid(xs)
        xs = [self.context_modules[i](xs[i]) for i in range(len(xs))]
        reg = torch.cat([self.heads_boxes[i](xs[i]) for i in range(len(xs))], dim=1)
        cls = torch.cat([self.heads_class[i](xs[i]) for i in range(len(xs))], dim=1)
             
        priors = get_priors(x.shape[2:], self.bases, dv, loc='center')
        scores = F.softmax(cls, dim=-1)[:, :, 1]
        b, s = get_results(reg, scores, priors, 0.5, 0.4, decode_mults=(0.1, 0.2))
        b, s = [[t.detach().cpu().numpy() for t in tl] for tl in [b, s]]
        return b, s   


class RetinaNet(nn.Module):

    def __init__(self, device='cpu'):
        super().__init__()
        backbone = ResNet50(retidx=[2, 3, 4], bn=(0.0, 'frozen'), num_freeze=5)
        cins = [512, 1024, 2048]
        cout = 256
        strides = [8, 16, 32, 64, 128]
        astarts = [32, 64, 128, 256, 512]
        anchors = make_anchors(astarts, [1, 2**(1/3), 2**(2/3)], [2, 1, 0.5], rounding=True)
        self.bases = list(zip(strides, anchors))
        anchors_per_level = len(anchors[0])
        self.num_classes = 91
        self.body = backbone
        self.fpn = FeaturePyramidNetwork(cins, cout, None, None, P6='fromP5', P7=True, smoothP5=True)
        self.cls_head = HeadShared(cout, anchors_per_level, self.num_classes)
        self.reg_head = HeadShared(cout, anchors_per_level, 4)
        self.to(device)
    
    def forward(self, imgs, targets=None):
        dv = next(self.parameters()).device
        x, sz_orig, sz_used = preprocess(imgs, dv, (800, 1333), 'torch')
        priors = get_priors(x.shape[2:], self.bases, dv, 'corner', 'fit')

        xs = self.body(x)
        xs = self.fpn(xs)
        reg = [self.reg_head(lvl) for lvl in xs]
        log = [self.cls_head(lvl) for lvl in xs]
        lvlen = [lvl.shape[1] for lvl in log]
        reg = torch.cat(reg, axis=1)
        log = torch.cat(log, axis=1)
        
        if self.training:
            gtb, gtl = prep_targets(targets, sz_used, sz_orig)
            return get_losses(gtb, gtl, priors, reg, log, (0.4, 0.5, True), None, ('focal', 'l1'), 'per_image', 'always_pos')
        else:
            scr = log.sigmoid_()
            b, s, c = self.postprocess(reg, scr, priors, lvlen, sz_used)
            b = scale_boxes(b, sz_orig, sz_used)
            b, s, c = [[t.detach().cpu().numpy() for t in tl] for tl in [b, s, c]]
            return b, s, c

    def postprocess(self, reg, scr, priors, lvlen, sz_used):
        n, dim, num_classes = scr.shape
        reg, scr = reg.reshape(-1, 4), scr.flatten()
        fidx = torch.nonzero(scr > 0.05).squeeze()
        fidx = top_per_level(fidx, scr, 1000, lvlen, n, mult=num_classes)
        scores = scr[fidx]
        classes = fidx % num_classes
        idx = torch.div(fidx, num_classes, rounding_mode='floor')
        imidx = idx.div(dim, rounding_mode='floor')
        
        boxes = decode_boxes(reg[idx], priors[idx % dim], mults=(1, 1), clamp=True)
        boxes = clamp_to_canvas(boxes, sz_used, imidx)
        b, s, c = final_nms(boxes, scores, classes, imidx, n, 0.5, 300)
        return b, s, c


class Detector_Retina():

    tvhub = 'https://download.pytorch.org/models/'
    links = {
        'coco_resnet50_tv': tvhub + 'retinanet_resnet50_fpn_coco-eeacb38b.pth',
        'wider_resnet50_b6': '14KX6VqF69MdSPk3Tr9PlDYbq7ArpdNUW',
        'wider_mobile_b6': '15zP8BP-5IvWXWZoYTNdvUJUiBqZ1hxu1',
        'mixed_resnet152_bbt': '1xB5RO99bVnXLYesnilzaZL2KWz4BsJfM',
        'mixed_resnet50_bbt': '1uraA7ZdCCmos0QSVR6CJgg0aSLtV4q4m',
        'wider_resnet50_bbt': '1pQLydyUUEwpEf06ElR2fw8_x2-P9RImT',
        'icartoon_resnet50_bbt': '12RsVC1QulqsSlsCleMkIYMHsAEwMyCw8'
    }

    def bbt_conversion(self, wd):
        # in this source, smoothP5 is not applied but the layer is still created for it for no reason
        for s in ['conv.weight', 'bn.weight', 'bn.bias', 'bn.running_mean',
                  'bn.running_var', 'bn.num_batches_tracked']:
            wd.pop('fpn.lateral_outs.3.' + s)
        # in this source, FPN extra P6 layer is placed between laterals and smooths, but we need after
        wl = list(wd.items())
        idx = [i for i, (n, _) in enumerate(wl) if n.startswith('fpn.lateral_ins.4')]
        els = [wl.pop(idx[0]) for _ in idx]
        pos = [i for i, (n, _) in enumerate(wl) if n.startswith('fpn.lateral_outs.')][-1]
        for el in els[::-1]:
            wl.insert(pos + 1, el)
        wd = dict(wl)
        return wd

    def __init__(self, src, device=None, train=False):
        assert src in self.links
        print('Initializing %s model for object detection' % 'Retina')
        dv = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')

        if src == 'coco_resnet50_tv':
            self.model = RetinaNet(dv)
        elif src == 'wider_resnet50_b6':
            self.model = RetinaFace_Biubug6(dv, False)
        elif src == 'wider_mobile_b6':
            self.model = RetinaFace_Biubug6(dv, True)
        elif src == 'mixed_resnet152_bbt':
            self.model = RetinaFace_BBT(dv, False)
        elif src == 'mixed_resnet50_bbt':
            self.model = RetinaFace_BBT(dv)
        elif src == 'wider_resnet50_bbt':
            self.model = RetinaFace_BBT(dv)
        elif src == 'icartoon_resnet50_bbt':
            self.model = RetinaFace_BBT(dv)

        wname = self.model.__class__.__name__.lower() + src
        wconv = None if not src.endswith('bbt') else self.bbt_conversion
        wbat = src == 'coco_resnet50_tv'
        load_weights(self.model, self.links[src], wname, wconv, add_num_batches=wbat)

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