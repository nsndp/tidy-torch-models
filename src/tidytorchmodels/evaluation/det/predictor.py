import math
import os.path as osp

import cv2
import numpy as np

from ...utils.image import resize_keep_ratio, pad_to_area, read_imsize_binary
from ...utils.pbar import tqdm


def get_predictions(load, model, fn, updir, imdir, mult, bs):
    """Runs inference with ``model`` on ``fn`` images from ``imdir`` folder and returns predictions,
    also saving them on disk to ``updir/pred_<set_name>_<model_class_name>.npy`` as one stacked array
    where every row is: "image_index, x1, y1, x2, y2, score" (dtype = np.float32).

    Alternatively, if ``load`` is specified, treats it as a path to a saved array in the same format,
    loads it and converts back into a list with len = image count (while other params are ignored).
    """
    if load is None:
        sn = osp.basename(updir)
        mn = model.__class__.__name__
        print('Getting predictions for the images using %s model' % mn)
        if bs == 1:
            preds = predict(fn, model, imdir)
        else:
            preds = predict_batch(fn, model, imdir, bs)
        fp = osp.join(updir, 'pred_%s_%s.npy' % (sn, mn))
        print('Saving predictions for possible repeated use to: %s' % fp)
        inds = [np.array([i] * len(p))[:, None] for i, p in enumerate(preds)]
        flat = np.hstack([np.concatenate(inds), np.concatenate(preds)])
        np.save(fp, flat)
    else:
        print('Loading saved predictions from: %s' % load)
        flat = np.load(load)
        inds = flat[:, 0].astype(np.int64)
        preds = np.split(flat[:, 1:], np.bincount(inds).cumsum()[:-1])
    return preds


def predict(fn, model, imdir):
    """Runs inference with ``model`` on every image separately."""
    preds = []
    for k in tqdm(range(len(fn))):
        img = cv2.imread(osp.join(imdir, fn[k]))
        pred = model([img])[0]
        #pred = pred[pred[:, 4].argsort()[::-1]].astype(np.float32)
        preds.append(pred)
    return preds


def predict_batch(fn, model, imdir, bs):
    """Runs inference with ``model`` in batches of size ``bs``. For that to work on a dataset
    where images can have all kinds of sizes and aspect ratios, sorts the images by ratio and area,
    then takes a size of a median image within each batch, then resizes to that (while keeping the
    ratio and padding the remainders with black bars). This aims to minimize the distortion
    compared to, say, just squashing all images to a single fixed size.

    The scales for each image are remembered and used to scale the predictions back to the original
    size, and the padding is always applied to the right and bottom so that it doesn't affect boxes'
    coordinates.
    """
    areas, ratios = [], []
    for f in fn:
        p = osp.join(imdir, f)
        w, h = read_imsize_binary(p)
        #im = cv2.imread(p)
        #assert w == im.shape[1], 'wrong in %s' % p
        #assert h == im.shape[0], 'wrong in %s' % p
        areas.append(w * h)
        ratios.append(round(w / h, 1))
    idx = np.lexsort((areas, ratios))
    
    res = []
    with tqdm(total=len(fn)) as pbar:
        for bn in range(math.ceil(len(fn) / bs)):
            bidx = idx[bs*bn:bs*(bn+1)]
            bfn = [fn[i] for i in bidx]
            imgs = [cv2.imread(osp.join(imdir, p)) for p in bfn]
            h, w = imgs[len(bidx) // 2].shape[:2]
            tups = [resize_keep_ratio(im, (w, h)) for im in imgs]
            imgs = [pad_to_area(im, (w, h)) for im, _ in tups]
            scls = [scl for _, scl in tups]
            preds = model(imgs)
            for i in range(len(preds)):
                preds[i][:, :4] /= scls[i]
            res.extend(preds)
            pbar.update(len(preds))
    res = [res[i] for i in np.argsort(idx)]
    return res