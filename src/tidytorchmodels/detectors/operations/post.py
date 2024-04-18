import numpy as np
import torch
import torchvision.ops

from .bbox import decode_boxes


def final_nms(boxes, scores, classes, imidx, n, thresh, imtop=None):
    res = []
    for i in range(n):
        bi, si, ci = [x[imidx == i] for x in [boxes, scores, classes]]
        keep = torchvision.ops.batched_nms(bi, si, ci, thresh)[:imtop]
        res.append((bi[keep], si[keep], ci[keep]))
    return map(list, zip(*res))
    #groups = imidx * 1000 + classes
    #keep = torchvision.ops.batched_nms(boxes, scores, groups, thresh)
    #if imtop:
    #    keep = torch.cat([keep[imidx[keep] == i][:imtop] for i in range(n)])
    #b, s, c, imidx = [x[keep] for x in [boxes, scores, classes, imidx]]
    #b, s, c = [[x[imidx == i] for i in range(n)] for x in [b, s, c]]
    #return b, s, c


def get_results(reg, scores, priors, score_thr, iou_thr, decode_mults):
    n, dim = reg.shape[:2]
    reg, scores = reg.reshape(-1, 4), scores.flatten()
    idx = torch.nonzero(scores > score_thr).squeeze()
    scores = scores[idx]
    imidx = idx.div(dim, rounding_mode='floor')
    boxes = decode_boxes(reg[idx], priors[idx % dim], mults=decode_mults)
    res = []
    for i in range(n):
        bi, si = boxes[imidx == i], scores[imidx == i]
        keep = torchvision.ops.nms(bi, si, iou_thr)
        res.append((bi[keep], si[keep]))
    return map(list, zip(*res))


def get_lvidx(idx, lvsizes):
    """
    dim = 8000
    lsz = [3500, 2500, 1000, 600, 400]
    idx = torch.tensor([0, 5999, 6000, 7999, 8000, 15000, 20000])

    res = tensor([0, 1, 2, 4, 0, 3, 1])
    cumsum = tensor([3500, 6000, 7000, 7600, 8000])
    idx % dim = tensor([   0, 5999, 6000, 7999,    0, 7000, 4000])
    """
    boundaries = torch.tensor(lvsizes).to(idx.device).cumsum(0)
    return torch.bucketize(idx, boundaries, right=True)
    #return torch.gt(boundaries, idx[:, None]).to(dtype=int).argmax(dim=1)


def top_per_level(idx, scores, limit, lvlen, nimg, mult=1):
    """"""
    if not limit:
        return idx
    sel = []
    for i in range(nimg):
        borders = np.cumsum([0] + lvlen)
        borders += i * sum(lvlen)
        borders *= mult
        for j in range(1, len(borders)):
            lvidx = idx[(idx >= borders[j - 1]) * (idx < borders[j])]
            if lvidx.shape[0] > limit:
                _, top = torch.topk(scores[lvidx], min(limit, lvidx.shape[0]))
                sel.append(lvidx[top])
            else:
                sel.append(lvidx)
    return torch.cat(sel)


def top_per_class(scores, classes, imidx, limit):
    """"""
    sel = []
    for c in classes.unique():
        for i in range(torch.max(imidx) + 1):
            cidx = torch.nonzero((classes == c) * (imidx == i)).squeeze(-1)
            if cidx.numel() > limit:
                _, top = torch.topk(scores[cidx], limit)
                sel.append(cidx[top])
            elif cidx.numel() > 0:
                sel.append(cidx)
    return torch.cat(sel)


# main source: https://github.com/rbgirshick/fast-rcnn/blob/master/lib/utils/nms.py
# batch (see coordinate trick): https://pytorch.org/vision/stable/_modules/torchvision/ops/boxes.html
# on adding +1 to area calculation or not: https://stackoverflow.com/a/51730512/8874388
# (torchvision.ops.nms/batched_nms don't have +1)
# on speed in numpy vs torch: https://discuss.pytorch.org/t/nms-implementation-slower-in-pytorch-compared-to-numpy/36665/7
def nms(boxes, scores, thresh):
    x1, x2 = boxes[:, 0], boxes[:, 2]
    y1, y2 = boxes[:, 1], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    keep = []
    order = scores.argsort()[::-1]
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ious = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.nonzero(ious <= thresh)[0]
        order = order[inds + 1]
    return keep