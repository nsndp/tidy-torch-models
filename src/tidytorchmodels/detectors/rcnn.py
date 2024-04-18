from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops

from ..backbones.basic import ConvUnit
from ..backbones.resnet import ResNet50
from ..backbones.mobilenet import MobileNetV3L
from .components.fpn import FeaturePyramidNetwork
from .operations.anchor import get_priors, make_anchors
from .operations.bbox import clamp_to_canvas, convert_to_cwh, decode_boxes, remove_small, scale_boxes
from .operations.loss import calc_losses, get_losses, match_with_targets
from .operations.post import get_lvidx, final_nms
from .operations.prep import preprocess, prep_targets
from .operations.roi import roi_align_multilevel
from ..utils.weights import load_weights


class RegionProposalNetwork(nn.Module):

    def __init__(self, c, num_anchors, conv_depth):
        super().__init__()
        self.conv = nn.Sequential(*[ConvUnit(c, c, 3, 1, 1, 'relu', None) for _ in range(conv_depth)])
        self.log = nn.Conv2d(c, num_anchors, 1, 1)
        self.reg = nn.Conv2d(c, num_anchors * 4, 1, 1)

    def head(self, x):
        n = x.shape[0]
        x = self.conv(x)
        reg = self.reg(x).permute(0, 2, 3, 1).reshape(n, -1, 4)
        log = self.log(x).permute(0, 2, 3, 1).reshape(n, -1, 1)
        return reg, log

    def filt_dec(self, regs, logs, priors, lvtop):
        res = []
        n = regs[0].shape[0]
        for reg, log, p in zip(regs, logs, priors):
            log, top = log.topk(min(lvtop, log.shape[1]), dim=1)
            reg = reg.gather(1, top.expand(-1, -1, 4))
            pri = p.expand(n, -1, -1).gather(1, top.expand(-1, -1, 4))
            boxes = decode_boxes(reg, pri, mults=(1, 1), clamp=True)
            res.append((boxes, log, log.shape[1]))
        return map(list, zip(*res))

    def forward(self, fmaps, priors, imsizes, settings, gtboxes=None):
        score_thr, iou_thr, imtop_infer, min_size, lvtop_infer, imtop_train, lvtop_train = settings
        lvtop = lvtop_infer if not self.training else lvtop_train
        imtop = imtop_infer if not self.training else imtop_train

        tuples = [self.head(x) for x in fmaps]
        regs, logs = map(list, zip(*tuples))
        
        dregs = [x.detach() for x in regs]
        dlogs = [x.detach() for x in logs]
        boxes, logits, lvlen = self.filt_dec(dregs, dlogs, priors, lvtop)
        boxes = torch.cat(boxes, axis=1)
        obj = torch.cat(logits, axis=1).sigmoid()

        n, dim = boxes.shape[:2]
        boxes, obj = boxes.reshape(-1, 4), obj.flatten()
        idx = torch.nonzero(obj >= score_thr).squeeze()
        boxes, obj = boxes[idx], obj[idx]
        imidx = idx.div(dim, rounding_mode='floor')

        boxes = clamp_to_canvas(boxes, imsizes, imidx)
        boxes, obj, idx, imidx = remove_small(boxes, min_size, obj, idx, imidx)
        groups = imidx * 10 + get_lvidx(idx % dim, lvlen)
        keep = torchvision.ops.batched_nms(boxes, obj, groups, iou_thr)
        keep = torch.cat([keep[imidx[keep] == i][:imtop] for i in range(n)])
        boxes, imidx = boxes[keep], imidx[keep]
        
        if not self.training:
            return boxes, imidx
        else:
            regs = torch.cat(regs, axis=1)
            logs = torch.cat(logs, axis=1)
            priors = torch.cat(priors)
            loss_obj, loss_reg = get_losses(gtboxes, None, priors, regs, logs, matcher=(0.3, 0.7, True),
                                            sampler=(256, 0.5), types=('ce_bin', 'l1_s'), avg_divs='always_all')
            return boxes, imidx, (loss_obj, loss_reg)


class RoIProcessingNetwork(nn.Module):

    def __init__(self, c, roi_map_size, clin, roi_convdepth, roi_mlp_depth,
                 num_classes, bckg_class_first, roialign_settings):
        super().__init__()
        self.conv = nn.ModuleList([ConvUnit(c, c, 3, 1, 1, 'relu') for _ in range(roi_convdepth)])
        c1 = c * roi_map_size ** 2
        self.fc = nn.ModuleList([nn.Linear(c1 if i == 0 else clin, clin) for i in range(roi_mlp_depth)])
        self.cls = nn.Linear(clin, 1 + num_classes)
        self.reg = nn.Linear(clin, num_classes * 4)
        self.ralign_set = roialign_settings
        self.bckg_first = bckg_class_first

    def heads(self, x):
        if self.conv:
            for layer in self.conv:
                x = layer(x)
        x = x.flatten(start_dim=1)
        if self.fc:
            for mlp in self.fc:
                x = F.relu(mlp(x))
        a = self.reg(x)
        b = self.cls(x)
        return a, b
    
    def forward(self, proposals, imidx, fmaps, fmaps_strides, imsizes, settings, gtb=None, gtl=None):
        if self.training:
            # to list
            proposals = [proposals[imidx == i] for i in range(len(gtb))]
            proposals = [torch.cat([p, b.to(torch.float32)]) for p, b in zip(proposals, gtb)]
            # main
            targets, labels, sidx = match_with_targets(gtb, gtl, proposals, 0.5, 0.5, False, 512, 0.25)
            proposals = [p[sampled] for p, sampled in zip(proposals, sidx)]
            # back to joined
            imidx = [torch.full([len(p)], i) for i, p in enumerate(proposals)]
            proposals, imidx = [torch.cat(x) for x in [proposals, imidx]]
        
        roi_maps = roi_align_multilevel(proposals, imidx, fmaps, fmaps_strides, self.ralign_set)
        reg, log = self.heads(roi_maps)
        reg = reg.reshape(reg.shape[0], -1, 4)

        if self.training:
            labels = torch.cat(labels)
            targets = torch.cat(targets)
            loss_cls = F.cross_entropy(log, labels)
            reg = reg[labels > 0, labels[labels > 0] - 1] # minus 1 because we removed the bckg class
            loss_reg = F.smooth_l1_loss(reg, targets, beta=1/9, reduction='sum') / labels.numel()
            return loss_cls, loss_reg
       
        scr = F.softmax(log, dim=-1)
        cls = torch.arange(log.shape[1], device=log.device).view(1, -1).expand_as(log)
        scr = scr[:, :-1] if not self.bckg_first else scr[:, 1:]
        cls = cls[:, :-1] if not self.bckg_first else cls[:, 1:]

        score_thr, iou_thr, imtop, min_size = settings

        n = torch.max(imidx).item() + 1
        dim = reg.shape[1]
        reg, scr, cls = reg.reshape(-1, 4), scr.flatten(), cls.flatten()
        fidx = torch.nonzero(scr > score_thr).squeeze()
        reg, scr, cls = reg[fidx], scr[fidx], cls[fidx]
        idx = fidx.div(dim, rounding_mode='floor')
        proposals, imidx = proposals[idx], imidx[idx]

        proposals = convert_to_cwh(proposals, in_place=True)
        boxes = decode_boxes(reg, proposals, mults=(0.1, 0.2), clamp=True)
        boxes = clamp_to_canvas(boxes, imsizes, imidx)
        boxes, scr, cls, imidx = remove_small(boxes, min_size, scr, cls, imidx)
        b, s, c = final_nms(boxes, scr, cls, imidx, n, iou_thr, imtop)
        return b, s, c


class FasterRCNN(nn.Module):

    def __init__(self, device='cpu',
                 # 1) backbone type and its batch norm settings
                 bbone='resnet50', bn_eps=1e-5, bn_freeze=False,
                 # 2) architectural settings for FPN and RPN
                 fpn_batchnorm=None, rpn_convdepth=1, roi_convdepth=0, roi_mlp_depth=2,
                 # 3) classes info and ROI algo details
                 num_classes=80, bckg_class_first=False, roialign_settings=(0, True),
                 # 4) settings for preprocessing, anchors and priors
                 prep_resize='cv2', priors_patches='as_is', round_anchors=False,
                 resize_min=800, resize_max=1333,
                 # 5) settings for filtering during RPN and ROI forwards
                 score_thr1=0.0, iou_thr1=0.7, imtop1=1000, min_size1=0, lvtop=1000,
                 score_thr2=0.05, iou_thr2=0.5, imtop2=100, min_size2=0,
                 imtop1_train=2000, lvtop_train=2000):
        super().__init__()

        bn = bn_eps if not bn_freeze else (bn_eps, 'frozen')
        if bbone == 'resnet50':
            self.body = ResNet50(bn=bn, num_freeze=2)
            cins, self.strides = [256, 512, 1024, 2048], [4, 8, 16, 32, 64]
            anchors = make_anchors([32, 64, 128, 256, 512], [1], [2, 1, 0.5], round_anchors)
        elif bbone == 'mobilenetv3l':
            self.body = MobileNetV3L([13, 16], bn=bn, num_freeze=7)
            cins, self.strides = [160, 960], [32, 32, 64]
            anchors = make_anchors([32, 32, 32], [1, 2, 4, 8, 16], [2, 1, 0.5], round_anchors)

        self.bases = list(zip(self.strides, anchors))
        self.resize = (resize_min, resize_max)
        self.resize_with = prep_resize
        self.priors_patches = priors_patches
        self.bxset1 = (score_thr1, iou_thr1, imtop1, min_size1, lvtop, imtop1_train, lvtop_train)
        self.bxset2 = (score_thr2, iou_thr2, imtop2, min_size2)

        self.fpn = FeaturePyramidNetwork(cins, 256, None, fpn_batchnorm, pool=True, smoothP5=True)
        self.rpn = RegionProposalNetwork(256, len(anchors[0]), rpn_convdepth)
        self.roi = RoIProcessingNetwork(256, 7, 1024, roi_convdepth, roi_mlp_depth,
                                        num_classes, bckg_class_first, roialign_settings)
        self.to(device)

    def forward(self, imgs, targets=None):
        dv = next(self.parameters()).device
        x, sz_orig, sz_used = preprocess(imgs, dv, self.resize, self.resize_with)
        priors = get_priors(x.shape[2:], self.bases, dv, 'corner', self.priors_patches, concat=False)
        xs = self.body(x)
        xs = self.fpn(xs)
        if self.training:
            gtb, gtl = prep_targets(targets, sz_used, sz_orig)
            p, imidx, p_losses = self.rpn(xs, priors, sz_used, self.bxset1, gtb)
            d_losses = self.roi(p, imidx, xs[:-1], self.strides[:-1], sz_used, self.bxset2, gtb, gtl)
            return (*p_losses, *d_losses)
        else:
            p, imidx = self.rpn(xs, priors, sz_used, self.bxset1)
            b, s, c = self.roi(p, imidx, xs[:-1], self.strides[:-1], sz_used, self.bxset2)
            b = scale_boxes(b, sz_orig, sz_used)
            b, s, c = [[t.detach().cpu().numpy() for t in tl] for tl in [b, s, c]]
            return b, s, c


class Detector_FasterRCNN():

    thub = 'https://download.pytorch.org/models/'
    mmhub = 'https://download.openmmlab.com/mmdetection/v2.0/'
    links = {
        'tv_resnet50_v1': thub + 'fasterrcnn_resnet50_fpn_coco-258fb6c6.pth',
        'tv_resnet50_v2': thub + 'fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth',
        'tv_mobilenetv3l_hires': thub + 'fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth',
        'tv_mobilenetv3l_lores': thub + 'fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth',
        'mm_resnet50': mmhub + 'faster_rcnn/faster_rcnn_r50_fpn_mstrain_3x_coco/'\
                               'faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth',
        'mm_resnet50_animefaces': 'https://github.com/hysts/anime-face-detector/'\
                                  'releases/download/v0.0.1/mmdet_anime-face_faster-rcnn.pth'
    }

    def tv_conversion(self, wd):
        # TorchVision's RoI head predicts boxes for background class too, only to discard
        # them right away, so might as well not calculate it in the first place
        nm = 'roi_heads.box_predictor.bbox_pred.'
        wd[nm + 'weight'] = wd[nm + 'weight'][4:, :] # [364, 1024] -> [360]
        wd[nm + 'bias'] = wd[nm + 'bias'][4:]        # [364] -> [360]
        return wd

    def mm_conversion(self, wd):
        # in MMDet weights for RoI head, representation FC and final reg/log FCs are switched over
        wl = list(wd.items())
        els = [wl.pop(-1) for _ in range(8)][::-1] # last 8 entries
        for el in els[4:] + els[:4]:
            wl.append(el)
        wd = dict(wl)
        return wd

    def __init__(self, src, device=None, train=False):
        assert src in self.links
        print('Initializing %s model for object detection' % 'FasterRCNN')
        dv = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')

        FasterRCNN_TorchVision = partial(FasterRCNN,
            num_classes=90, bckg_class_first=True, roialign_settings=(2, False),
            prep_resize='torch', priors_patches='fit', round_anchors=True,
            min_size1=1e-3, min_size2=1e-2)

        if src == 'mm_resnet50':
            self.model = FasterRCNN()
        elif src == 'mm_resnet50_animefaces':
            self.model = FasterRCNN(num_classes=1)
        elif src == 'tv_resnet50_v1':
            self.model = FasterRCNN_TorchVision(bn_eps=0.0, bn_freeze=True)
        elif src == 'tv_resnet50_v2':
            self.model = FasterRCNN_TorchVision(fpn_batchnorm=1e-5,
                         rpn_convdepth=2, roi_convdepth=4, roi_mlp_depth=1)
        elif src == 'tv_mobilenetv3l_hires':
            self.model = FasterRCNN_TorchVision(bbone='mobilenetv3l', bn_freeze=True)
        elif src == 'tv_mobilenetv3l_lores':
            self.model = FasterRCNN_TorchVision(bbone='mobilenetv3l', bn_freeze=True,
                         resize_min=320, resize_max=640, lvtop=150, imtop1=150, score_thr1=0.05)

        wconv = self.mm_conversion if src.startswith('mm') else self.tv_conversion
        wsub = 'state_dict' if src.startswith('mm') else None
        wnbatches = src.startswith('tv') and src != 'tv_resnet50_v2'
        load_weights(self.model, self.links[src], 'frcnn_' + src, wconv, wsub, wnbatches)

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