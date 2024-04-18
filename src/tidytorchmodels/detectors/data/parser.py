import json
import os.path as osp

import numpy as np


def get_coco_data(gtdir):
    with open(osp.join(gtdir, 'gt.json')) as f:
        d = json.load(f)
    ann = [('%012d.jpg' % int(k), v['boxes'], v['classes']) for k, v in d.items()]
    ann.sort(key=lambda tup: tup[0])
    fn, gt, cl = map(list, zip(*ann))
    gt = [np.array(e) for e in gt]
    cl = [np.array(e) for e in cl]
    for a in gt:
        a[:, 2:] += a[:, :2]
    return fn, gt, cl