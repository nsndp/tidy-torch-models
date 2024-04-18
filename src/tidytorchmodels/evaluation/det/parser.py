from glob import glob
import os.path as osp

import numpy as np

from . import extra


def get_set_data(set_name, gtdir):
    if set_name == 'FDDB':
        fn, gt = get_fddb_data(gtdir)
    elif set_name == 'ICARTOON':
        fn, gt = get_icartoon_data(gtdir)
    elif set_name == 'PIXIV2018':
        fn, gt = get_pixiv2018_data(gtdir)
    elif set_name == 'PIXIV2018_ORIG':
        fn, gt = extra.get_pixiv2018_data_ORIG(gtdir)
    return fn, gt


def get_pixiv2018_data(gtdir):
    """Parses annotations given in a single "boxes.txt" file in the form of:
    <filename.jpg>, <number of boxes>, <n lines with x1, y1, x2, y2>, repeat.
    """
    fn, gt = [], []
    with open(osp.join(gtdir, 'boxes.txt')) as f:
        fni = f.readline()
        while fni:
            gti = []
            c = int(f.readline().rstrip())
            for _ in range(c):
                box = [int(c) for c in f.readline().rstrip().split(' ')]
                gti.append(box)
            gt.append(gti)
            fn.append(fni.rstrip())
            fni = f.readline()
    gt = [np.array(e) for e in gt]
    return fn, gt


def get_icartoon_data(gtdir):
    """Parses iCartoonFace detval annotations, which are given as '<filename>,x1,y1,x2,y2,face'"""
    fn, gt = [], []
    with open(osp.join(gtdir, 'personai_icartoonface_detval.csv')) as f:
        for line in f:
            s = line.split(',')
            name, box = s[0], [int(e) for e in s[1:5]]
            if not fn or fn[-1] != name:
                fn.append(name)
                gt.append([box])
            else:
                gt[-1].append(box)
    gt = [np.array(e) for e in gt]
    return fn, gt


def get_wider_data(gtdir):
    """Parses WIDER Face annotations. They are in Matlab's .mat files, so depends on scipy to read that.

    Source format example:
    event[i]:       13--Interview
    file[i][j]:     13_Interview_Interview_On_Location_13_921
    easy[i]:        []
    medium[i][j]:   [2 3]
    hard[i][j]:     [1 2 3 4]
    facebox[i][j]:  [[200 392  21  26]
                     [660 389  26  32]
                     [753 393  27  34]
                     [935 388  18  29]]

    The split into 3 difficulty settings is implemented by giving subsets of indices for all boxes,
    e.g. for example above for "medium" only boxes 2, 3 need to be detected and are treated as
    "false negatives" if absent, while boxes 1, 4 are ignored (so if a model detects them, they
    don't count as "false positives" (thus not lowering precision), but also don't count as
    "true positives" (thus not improving recall).
    """
    from scipy.io import loadmat
    gt_mat = loadmat(osp.join(gtdir, 'wider_face_val.mat'))
    hard_mat = loadmat(osp.join(gtdir, 'wider_hard_val.mat'))
    medium_mat = loadmat(osp.join(gtdir, 'wider_medium_val.mat'))
    easy_mat = loadmat(osp.join(gtdir, 'wider_easy_val.mat'))
    
    facebox_list = gt_mat['face_bbx_list']
    event_list = gt_mat['event_list']
    file_list = gt_mat['file_list']

    fn, gt, st = [], [], []
    for i in range(len(event_list)):
        for j in range(len(file_list[i][0])):
            flname = osp.join(event_list[i][0][0], file_list[i][0][j][0][0]) + '.jpg'
            bboxes = facebox_list[i][0][j][0]
            bboxes[:, 2] = bboxes[:, 2] + bboxes[:, 0]
            bboxes[:, 3] = bboxes[:, 3] + bboxes[:, 1]
            idx = [
                easy_mat['gt_list'][i][0][j][0],
                medium_mat['gt_list'][i][0][j][0],
                hard_mat['gt_list'][i][0][j][0]
            ]
            idx = [k if k.shape != (0, 0) else np.empty((0, 1), np.uint8) for k in idx]
            idx = [np.squeeze(k, axis=1) - 1 for k in idx]
            fn.append(flname)
            gt.append(bboxes)
            st.append(idx)
    return fn, gt, st


def get_fddb_data(gtdir):
    """Parses FDDB annotations. They are given as ellipses instead of boxes, so also converts that.
    
    Source line format: "major_axis_radius minor_axis_radius angle center_x center_y  1"
    (yes, always two spaces and 1 at the end)
    
    Also, here's a code example to draw an ellipse with cv2 if some testing is needed:
      center = (int(e[3]), int(e[4]))
      axesln = (int(e[0]), int(e[1]))
      angle = e[2] * 180 / np.pi
      cv2.ellipse(im, center, axesln, angle, 0, 360, (0, 0, 255), 2)
    """
    listing = sorted(glob(osp.join(gtdir, 'FDDB-fold-[0-9][0-9].txt')))
    details = sorted(glob(osp.join(gtdir, 'FDDB-fold-[0-9][0-9]-ellipseList.txt')))
    fn, gt = [], []
    for c in range(len(listing)):
        with open(listing[c]) as f:
            filenames = f.read().splitlines()
        with open(details[c]) as f:
            txt = f.read().splitlines()
        i = 0
        for f in filenames:
            assert f == txt[i]
            n = int(txt[i + 1])
            i += 2
            v = [list(map(float, s[:-3].split(' '))) for s in txt[i:i+n]]
            v = [get_ellipse_bounding_box(e[3], e[4], e[0], e[1], e[2]) for e in v]
            fn.append(f + '.jpg')
            gt.append(np.array(v))
            i += n
    return fn, gt


def get_ellipse_bounding_box(center_x, center_y, radius_x, radius_y, angle_radians):
    """Returns the bounding box for an arbitrarily rotated ellipse
    using a solution from here: https://stackoverflow.com/a/14163413
    """
    ux = radius_x * np.cos(angle_radians)
    uy = radius_x * np.sin(angle_radians)
    vx = radius_y * np.sin(angle_radians)
    vy = radius_y * np.cos(angle_radians)
    half_w = np.sqrt(ux * ux + vx * vx)
    half_h = np.sqrt(uy * uy + vy * vy)
    x1, x2 = center_x - half_w, center_x + half_w
    y1, y2 = center_y - half_h, center_y + half_h
    return [x1, y1, x2, y2]