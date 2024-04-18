import numpy as np
import torch

from ...utils.download import url_download

class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush']

unused_idx = [11, 25, 28, 29, 44, 65, 67, 68, 70, 82, 90]

def weights_91_to_80(t):
    idx = np.delete(np.arange(91), unused_idx) + 1
    idx = torch.tensor(idx)
    dims = t.shape[1:]
    z = t.view(-1, 91, *dims)
    t = torch.cat([z[:, idx], z[:, 0:1]], dim=1).reshape(-1, *dims)
    return t

def idx_91_to_80(a91):
    bins91 = np.array(unused_idx)
    a80 = a91 - np.digitize(a91, bins91)
    return a80

def idx_80_to_91(a80):
    bins80 = np.array(unused_idx) - np.arange(len(unused_idx))
    a91 = a80 + np.digitize(a80, bins80)
    return a91


def convert_official_annotations():
    import os
    import shutil
    url_download('http://images.cocodataset.org/annotations/annotations_trainval2017.zip')
    shutil.unpack_archive('annotations_trainval2017.zip', 'coco_tmp')
    os.remove('annotations_trainval2017.zip')  
    _make_gt_file('val')
    _make_gt_file('train')
    shutil.rmtree('coco_tmp')

def _make_gt_file(set_type):
    import json
    import os.path as osp
    with open(osp.join('coco_tmp', 'annotations', 'instances_' + set_type + '2017.json')) as f:
        d = json.load(f)
    gt = {}
    for a in d['annotations']:
        imid = a['image_id']
        if imid not in gt:
            gt[imid] = {}
            gt[imid]['boxes'] = []
            gt[imid]['classes'] = []
        gt[imid]['boxes'].append(a['bbox'])
        gt[imid]['classes'].append(a['category_id'])
    with open('coco_2017_' + set_type + '_bbox_gt.json', 'w') as f:
        json.dump(gt, f)