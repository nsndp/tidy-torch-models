import os.path as osp

import cv2
import numpy as np
import PIL.Image
import torch


def get_images(pattern, elements, lib='cv2'):
    testdir = osp.dirname(osp.realpath(__file__))
    paths = [osp.join(testdir, 'images', pattern % el) for el in elements]
    return [PIL.Image.open(pt) if lib == 'PIL' else cv2.imread(pt) for pt in paths]


def pairwise_cosine_distances(m):
    # same as (but with no sklearn dependency):
    #from sklearn.metrics.pairwise import cosine_distances
    #return cosine_distances(m)
    m_normalized = m / np.linalg.norm(m, axis=1).reshape(-1, 1)
    # equivalent:  m / np.sqrt((m * m).sum(axis=1, keepdims=True))
    dist = 1 - m_normalized @ m_normalized.T
    np.fill_diagonal(dist, 0)
    return dist


def run_training_coco(model, seed=None):
    imgs = get_images('coco_val2017_%s.jpg', ['000139', '455157'])
    gt1 = np.array([[236.98, 142.51, 261.68, 212.01], [  7.03, 167.76, 156.35, 262.63],
                    [557.21, 209.19, 638.56, 287.92], [358.98, 218.05, 414.98, 320.88],
                    [290.69, 218.  , 352.52, 316.48], [413.2 , 223.01, 443.37, 304.37],
                    [317.4 , 219.24, 338.98, 230.83], [412.8 , 157.61, 465.85, 295.62],
                    [384.43, 172.21, 399.55, 207.95], [512.22, 205.75, 526.96, 221.72],
                    [493.1 , 174.34, 513.39, 282.65], [604.77, 305.89, 619.11, 351.6 ],
                    [613.24, 308.24, 626.12, 354.68], [447.77, 121.12, 461.74, 143.  ],
                    [549.06, 309.43, 585.74, 399.1 ], [350.76, 208.84, 362.13, 231.39],
                    [412.25, 219.02, 421.88, 231.54], [241.24, 194.99, 255.46, 212.62],
                    [336.79, 199.5 , 346.52, 216.23], [321.21, 231.22, 446.77, 320.15]])
    gt2 = np.array([[243.83, 135.79, 446.04, 308.57], [286.2 , 286.2 , 589.66, 526.38],
                    [159.13, 158.14, 332.87, 504.14], [210.18, 365.73, 459.96, 564.18],
                    [353.8 , 329.35, 634.25, 496.18], [547.34, 322.7 , 640.  , 370.74],
                    [530.89, 257.95, 640.  , 326.39], [274.72, 272.47, 341.06, 299.12]])
    cl1 = np.array([64, 72, 72, 62, 62, 62, 62, 1, 1, 78, 82, 84, 84, 85, 86, 86, 62, 86, 86, 67])
    cl2 = np.array([28, 67, 1, 15, 15, 15, 67, 73])
    targ = ([gt1, gt2], [cl1, cl2])
    
    # since we're only checking loss values and not backproping,
    # we can disable gradients to save up on memory usage
    with torch.no_grad():
        ret = model(imgs, targ, seed=seed)
    losses = [r.item() for r in ret]
    return losses