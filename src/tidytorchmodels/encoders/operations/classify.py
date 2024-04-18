import os.path as osp
import numpy as np


def predict(dataset, probs, topk=5):
    assert dataset in ['ImageNet1K', 'Danbooru']
    home = osp.dirname(osp.dirname(osp.realpath(__file__)))
    
    if dataset == 'ImageNet1K':
        # https://github.com/pytorch/vision/blob/main/torchvision/models/_meta.py#L7
        clsf = osp.join(home, 'classes', 'imagenet1k.txt')
        with open(clsf, encoding='utf-8') as f:
            classes = [l for l in f.read().splitlines()]
    
    elif dataset == 'Danbooru':
        # https://raw.githubusercontent.com/arkel23/Danbooru2018AnimeCharacterRecognitionDataset_Revamped/master/labels/classid_classname.csv
        clsf = osp.join(home, 'classes', 'dafre.csv')
        with open(clsf, encoding='utf-8') as f:
            classes = [l.split(',') for l in f.read().splitlines()]
            classes = dict([(int(ind), nm) for ind, nm in classes])

    res = []
    for pb in probs:
        # https://stackoverflow.com/a/38772601
        idx = np.argpartition(pb, -topk)[-topk:]    # returns largest k unordered
        idx = idx[np.argsort(pb[idx])[::-1]]        # makes them ordered
        res.append([(classes[ind], pb[ind] * 100) for ind in idx])
    return res