import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbones.basic import ConvUnit
from ..backbones.mobilenet import MobileNetV2
from .operations.anchor import get_priors
from .operations.bbox import decode_boxes, scale_boxes
from .operations.post import get_lvidx, final_nms
from .operations.prep import preprocess
from ..utils.weights import load_weights

# adapted from: https://github.com/open-mmlab/mmdetection
# https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/dense_heads/yolo_head.py
# https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/post_processing/bbox_nms.py


def conv_unit(cin, cout, k, s=1):
    return ConvUnit(cin, cout, k, s, p=(k-1)//2, activ='lrelu_0.1')


class ResBlock(nn.Module):

    def __init__(self, c):
        super().__init__()
        self.conv1 = conv_unit(c, c // 2, k=1)
        self.conv2 = conv_unit(c // 2, c, k=3)
  
    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        return y + x


class Darknet53(nn.Module):

    def __init__(self, ):
        super().__init__()
        L, C = [1, 2, 8, 8, 4], [(32, 64), (64, 128), (128, 256), (256, 512), (512, 1024)]
        self.conv1 = conv_unit(3, 32, k=3)
        for i in range(len(L)):
            block = nn.Sequential()
            block.add_module('conv', conv_unit(C[i][0], C[i][1], k=3, s=2))
            for j in range(L[i]):
                block.add_module(f'res{j}', ResBlock(C[i][1]))
            self.add_module(f'conv_res_block{i + 1}', block)
  
    def forward(self, x):
        x = self.conv1(x)
        x1 = self.conv_res_block1(x)
        x2 = self.conv_res_block2(x1)
        x3 = self.conv_res_block3(x2)
        x4 = self.conv_res_block4(x3)
        x5 = self.conv_res_block5(x4)
        return (x3, x4, x5)
        

class DetectionBlock(nn.Module):

    def __init__(self, cin, cout):
        super().__init__()
        self.layers = nn.Sequential(
            conv_unit(cin, cout, k=1),
            conv_unit(cout, cout*2, k=3),
            conv_unit(cout*2, cout, k=1),
            conv_unit(cout, cout*2, k=3),
            conv_unit(cout*2, cout, k=1)
        )
  
    def forward(self, x):
        return self.layers(x)
        

class YOLOv3Neck(nn.Module):

    def __init__(self, cin, cout):
        super().__init__()
        self.detect1 = DetectionBlock(cin[2], cout[2])
        self.conv1 = conv_unit(cout[2], cout[1], k=1)
        self.detect2 = DetectionBlock(cin[1] + cout[1], cout[1])
        self.conv2 = conv_unit(cout[1], cout[0], k=1)
        self.detect3 = DetectionBlock(cin[0] + cout[0], cout[0])
  
    def forward(self, x):
        (x1, x2, x3) = x
        y3 = self.detect1(x3)
        t = self.conv1(y3)
        t = F.interpolate(t, scale_factor=2)
        t = torch.cat((t, x2), dim=1)
        y2 = self.detect2(t)
        t = self.conv2(y2)
        t = F.interpolate(t, scale_factor=2)
        t = torch.cat((t, x1), dim=1)
        y1 = self.detect3(t)
        return (y3, y2, y1)
        

class YOLOv3Head(nn.Module):

    def __init__(self, cin, cmid, num_classes):
        super().__init__()
        self.convs_bridge = nn.ModuleList([
            conv_unit(cin[2], cmid[2], k=3),
            conv_unit(cin[1], cmid[1], k=3),
            conv_unit(cin[0], cmid[0], k=3)
        ])
        cout = (num_classes + 5) * 3
        self.convs_pred = nn.ModuleList([
            nn.Conv2d(cmid[2], cout, 1),
            nn.Conv2d(cmid[1], cout, 1),
            nn.Conv2d(cmid[0], cout, 1)
        ])
  
    def forward(self, x):
        maps = []
        for i in range(3):
            y = x[i]
            y = self.convs_bridge[i](y)
            y = self.convs_pred[i](y)
            maps.append(y)
        return tuple(maps)


class YOLOv3(nn.Module):

    # https://becominghuman.ai/understanding-anchors-backbone-of-object-detection-using-yolo-54962f00fbbb

    bases_def = [
        (32, [(116, 90), (156, 198), (373, 326)]),
        (16, [(30, 61), (62, 45), (59, 119)]),
        (8,  [(10, 13), (16, 30), (33, 23)])
    ]
    bases_alt = [
        (32, [(220, 125), (128, 222), (264, 266)]),
        (16, [(35, 87), (102, 96), (60, 170)]),
        (8,  [(10, 15), (24, 36), (72, 42)])
    ]

    def __init__(self, device='cpu', backbone='darknet', canvas_size=608, num_classes=1, alt_bases=False):
        super().__init__()
        assert backbone in ['darknet', 'mobile2']
        self.canvas_size = canvas_size
        self.num_classes = num_classes
        self.bases = YOLOv3.bases_def if not alt_bases else YOLOv3.bases_alt
        if backbone == 'darknet':
            self.backbone = Darknet53()
            cbone, cneck, chead = [256, 512, 1024], [128, 256, 512], [256, 512, 1024]
            self.norm = {'means': None, 'stdvs': 255}
        elif backbone == 'mobile2':
            self.backbone = MobileNetV2([3, 5, 7])
            cbone, cneck, chead = [32, 96, 320], [96] * 3, [96] * 3
            self.norm = {'means': 'imagenet', 'stdvs': 'imagenet'}
        self.neck = YOLOv3Neck(cbone, cneck)
        self.head = YOLOv3Head(cneck, chead, num_classes)
        self.to(device)
  
    def forward(self, imgs):
        dv = next(self.parameters()).device
        x, szo, sz = preprocess(imgs, dv, self.canvas_size, 'cv2', **self.norm)
        xs = self.backbone(x)
        xs = self.neck(xs)
        xs = self.head(xs)
        priors = get_priors(x.shape[-2:], self.bases, dv, 'center')
        b, s, c = self.postprocess(xs, priors, self.num_classes)
        b = scale_boxes(b, szo, sz)
        b, s, c = [[t.detach().cpu().numpy() for t in tl] for tl in [b, s, c]]
        return b, s, c

    def postprocess(self, pred_maps, priors, num_classes):
        maps = [m.permute(0, 2, 3, 1).reshape(m.shape[0], -1, num_classes + 5) for m in pred_maps]
        map_sizes = [m.shape[1] for m in maps]
        maps = torch.cat(maps, dim=1)
        reg = maps[..., :4]
        obj = maps[..., 4].sigmoid()
        scr = maps[..., 5:].sigmoid()
        
        n, dim, num_classes = scr.shape
        reg, scr, obj = reg.reshape(-1, 4), scr.reshape(-1, num_classes), obj.flatten()
        oidx = torch.nonzero(obj >= 0.005).squeeze()
        scr, obj = scr[oidx], obj[oidx]
        s = scr.flatten()
        fidx = torch.nonzero(s > 0.05).squeeze()
        idx = torch.div(fidx, num_classes, rounding_mode='floor')
        s = s[fidx] * obj[idx]
        c = fidx % num_classes
        idx = oidx[idx]
        imidx = idx.div(dim, rounding_mode='floor')

        strides = [bs[0] for bs in self.bases]
        lvidx = get_lvidx(idx % dim, map_sizes)
        stidx = torch.tensor(strides)[lvidx].unsqueeze(-1)
        b = decode_boxes(reg[idx], priors[idx % dim], mode='yolo', strides=stidx)
        b, s, c = final_nms(b, s, c, imidx, n, 0.45, 100)
        return b, s, c


class Detector_YOLOv3():

    mmhub = 'https://download.openmmlab.com/mmdetection/v2.0/yolo/'
    links = {
        'anime': 'https://github.com/hysts/anime-face-detector/'\
                 'releases/download/v0.0.1/mmdet_anime-face_yolov3.pth',
        'wider': '1pjg1_IeAuzgRzZiY92r71uzd_amfcegu',
        'coco_darknet': mmhub + 'yolov3_d53_mstrain-608_273e_coco/'\
                        'yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth',
        'coco_mobile2_416': mmhub + 'yolov3_mobilenetv2_mstrain-416_300e_coco/'\
                            'yolov3_mobilenetv2_mstrain-416_300e_coco_20210718_010823-f68a07b3.pth',
        'coco_mobile2_320': mmhub + 'yolov3_mobilenetv2_320_300e_coco/'\
                            'yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth'
    }

    def __init__(self, src, device=None, train=False):
        assert src in self.links
        print('Initializing %s model for object detection' % 'YOLOv3')
        dv = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')

        if src == 'anime':
            self.model = YOLOv3(dv)
        elif src == 'wider':
            self.model = YOLOv3(dv)
        elif src == 'coco_darknet':
            self.model = YOLOv3(dv, num_classes=80)
        elif src == 'coco_mobile2_416':
            self.model = YOLOv3(dv, 'mobile2', 416, 80)
        elif src == 'coco_mobile2_320':
            self.model = YOLOv3(dv, 'mobile2', 320, 80, alt_bases=True)
        
        wsub = None if src == 'wider' else 'state_dict'
        load_weights(self.model, self.links[src], 'yolov3_' + src, sub=wsub)

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