import math

import torch


def make_anchors(dims, scales=[1], ratios=[1], rounding=False):
    """For every possible combination (D, S, R) of dims, scales and ratios,
    makes a box with area = D*D*S*S and aspect ratio = R,
    returning len(dims) lists, each with len(scales) * len(ratios) tuples.

    Example: make_anchors([16, 32], scales=[1, 0.5, 0.1], ratios=[1, 2])
    Output: [[(16, 16), (8, 8), (1.6, 1.6), (22.63, 11.31), (11.31, 5.66), (2.26, 1.13)],
             [(32, 32), (16, 16), (3.2, 3.2), (45.25, 22.63), (22.63, 11.31), (4.53, 2.26)]]
    """
    if rounding:
        return _make_anchors_rounded(dims, scales, ratios)
    mult = [math.sqrt(ar) for ar in ratios]
    anchors = [[(d * s * m, d * s / m) for m in mult for s in scales] for d in dims]
    return anchors


def _make_anchors_rounded(dims, scales, ratios):
    """Same as make_anchors but with two intermediate roundings to replicate TorchVision code:
    https://github.com/pytorch/vision/blob/main/torchvision/models/detection/anchor_utils.py#L58
    https://github.com/pytorch/vision/blob/main/torchvision/models/detection/retinanet.py#L51
    """
    mult = [math.sqrt(ar) for ar in ratios]
    dims_scaled = [[int(d * s) for s in scales] for d in dims]
    anchors = [[(int(d * s) * m, int(d * s) / m) for m in mult for s in scales] for d in dims]
    anchors = [[(round(w / 2) * 2, round(h / 2) * 2) for (w, h) in group] for group in anchors]
    return anchors


def get_priors(img_size, bases, dv='cpu', loc='center', patches='as_is', concat=True):
    """For every (stride, anchors) pair in ``bases`` list, walk through every stride-sized
    square patch of ``img_size`` canvas left-right, top-bottom and return anchors-sized boxes
    drawn around each patch's center in a form of (center_x, center_y, width, height).
    
    Example: get_priors((90, 64), [(32, [(8, 4), (25, 15)])])
    Output: shape = (12, 4)
    [[16, 16, 8, 4], [16, 16, 25, 15], [48, 16, 8, 4], [48, 16, 25, 15],
     [16, 48, 8, 4], [16, 48, 25, 15], [48, 48, 8, 4], [48, 48, 25, 15],
     [16, 80, 8, 4], [16, 80, 25, 15], [48, 80, 8, 4], [48, 80, 25, 15]]

    In case of square anchors, only one dimension can be provided, i.e. [(8, [16, 32])]
    will be automatically turned into [(8, [(16, 16), (32, 32)])].

    If loc='corner', then boxes are drawn around patches' top-left corners instead of centers.

    If patches='fit', then stride is adjusted so that patches fit the canvas without 'going over'.
    E.g. for canvas (h=800, w=1216) and stride 32, ``patches`` param will have no effect (since
    such canvas can be divided into 32x32 areas perfectly), but for stride 128, patches will
    change from 128x128 to 114x121 if the param = 'fit'.
    """
    assert loc in ['center', 'corner']
    assert patches in ['as_is', 'fit']
    p = []
    h, w = img_size
    if isinstance(bases[0][1][0], int):
        bases = [(s, [(a, a) for a in l]) for (s, l) in bases]
    for stride, anchors in bases:
        nx = math.ceil(w / stride)
        ny = math.ceil(h / stride)
        step_x = stride if patches == 'as_is' else w // nx
        step_y = stride if patches == 'as_is' else h // ny
        xs = torch.arange(nx, device=dv, dtype=torch.float32) * step_x
        ys = torch.arange(ny, device=dv, dtype=torch.float32) * step_y
        if loc == 'center':
            xs += step_x / 2
            ys += step_y / 2
        c = torch.dstack(torch.meshgrid(xs, ys, indexing='xy')).reshape(-1, 2)
        # could replace line above by "torch.cartesian_prod(xs, ys)" but that'd be for indexing='ij'
        c = c.repeat_interleave(len(anchors), dim=0)
        s = torch.tensor(anchors, device=dv, dtype=torch.float32).repeat(nx*ny, 1)
        p.append(torch.hstack([c, s]))
    if not concat:
        return p
    return torch.cat(p)