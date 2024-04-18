import math

import torch


def decode_boxes(pred, priors, mults=(1, 1), clamp=False, mode='rcnn', strides=None):
    """Converts predicted boxes from network outputs into actual image coordinates based on some
    fixed starting ``priors`` using Eq.1-4 from here: https://arxiv.org/pdf/1311.2524.pdf
    (as linked by Fast R-CNN paper, which is in turn linked by RetinaFace paper).

    Multipliers 0.1 and 0.2 are often referred to as "variances" in various implementations and used
    for normalizing/numerical stability purposes when encoding boxes for training (and thus are needed
    here too for scaling the numbers back). See https://github.com/rykov8/ssd_keras/issues/53 and
    https://leimao.github.io/blog/Bounding-Box-Encoding-Decoding/#Representation-Encoding-With-Variance
    """
    assert mode in ['rcnn', 'yolo']
    if mode == 'yolo':
        assert mults == (1, 1), strides is not None
    mult_xy, mult_wh = mults
    max_exp_input = math.log(1000 / 16) if clamp else None
    if mode == 'rcnn':
        xys = priors[..., 2:] * mult_xy * pred[..., :2] + priors[..., :2]
    elif mode == 'yolo':
        xys = strides * (pred[..., :2].sigmoid() - 0.5) + priors[..., :2]
    whs = priors[..., 2:] * exp_clamped(mult_wh * pred[..., 2:], max_exp_input)
    boxes = torch.cat([xys - whs / 2, xys + whs / 2], dim=-1)
    return boxes


def exp_clamped(x, max_=None):
    if not max_:
        return torch.exp(x)
    else:
        return torch.exp(torch.clamp(x, max=max_))


def encode_boxes(boxes, priors, mults):
    """boxes - (x1, y1, x2, y2), priors - (cx, cy, w, h)"""
    boxes = convert_to_cwh(boxes)
    rel_xys = (boxes[..., :2] - priors[..., :2]) / priors[..., 2:] / mults[0]
    rel_whs = torch.log(boxes[..., 2:] / priors[..., 2:]) / mults[1]
    return torch.cat([rel_xys, rel_whs], dim=-1)


def convert_to_cwh(boxes, in_place=False):
    """from (x1, y1, x2, y2) to (cx, cy, w, h)"""
    ret = boxes if in_place else boxes.clone()
    ret[..., 2:] -= ret[..., :2]
    ret[..., :2] += ret[..., 2:] * 0.5
    return ret


def convert_to_xyxy(boxes, in_place=False):
    """from (cx, cy, w, h) to (x1, y1, x2, y2)"""
    ret = boxes if in_place else boxes.clone()
    ret[..., :2] -= ret[..., 2:] * 0.5
    ret[..., 2:] += ret[..., :2]
    return ret


def clamp_to_canvas(boxes, imsizes, imidx):
    dv = imidx.device
    mx = torch.tensor(imsizes).to(dv).flip(1).repeat(1, 2)[imidx, :]
    boxes.clamp_(min=torch.tensor(0).to(dv), max=mx)
    return boxes


def remove_small(boxes, min_size, *args):
    boxes = boxes.view(-1, 4)
    ws = boxes[:, 2] - boxes[:, 0]
    hs = boxes[:, 3] - boxes[:, 1]
    mask = (ws > min_size) & (hs > min_size)
    #keep = (boxes[:, 2:] - boxes[:, :2] >= min_size).all(dim=1)
    if torch.count_nonzero(mask) < boxes.shape[0]:
        boxes = boxes[mask]
        args = [t[mask] if (t is not None) else None for t in args]
    return [boxes.squeeze(), *args]


def scale_boxes(boxes, target_imsizes, current_imsizes):
    scales = torch.tensor(target_imsizes) / torch.tensor(current_imsizes)
    scales = scales.to(boxes[0].device).flip(1).repeat(1, 2)
    boxes = [boxes[i] * scales[i] for i in range(len(boxes))]
    return boxes


def calc_iou_matrix(a, b, plus1=False):
    """Calculates intersection over union for every pair of two groups of boxes.
    I.e. if a.shape = (n, 4) and b.shape = (m, 4), then result.shape = (n, m),
    and result[i][j] = IoU between box a[i] and box b[j].

    Algorithm: https://stackoverflow.com/a/42874377
    Implementation: https://github.com/pytorch/vision/blob/main/torchvision/ops/boxes.py#L255
    On doing +1 or not: https://stackoverflow.com/a/51730512/8874388

    In general, doing x[:, None] + y allows you to get pairwise results thanks to
    automatic broadcasting: (N,1,D) + (M,D) <=> (N,1,D) + (1,M,D) = (N,M,D)
    """
    p = 0 if not plus1 else 1
    area1 = (a[:, 2] - a[:, 0] + p) * (a[:, 3] - a[:, 1] + p)
    area2 = (b[:, 2] - b[:, 0] + p) * (b[:, 3] - b[:, 1] + p)
    lt = torch.maximum(a[:, None, :2], b[:, :2])
    rb = torch.minimum(a[:, None, 2:], b[:, 2:])
    wh = (rb - lt + p).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / union