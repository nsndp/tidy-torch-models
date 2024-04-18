import numpy as np

from ...utils.pbar import tqdm

# sources:
# https://github.com/wondervictor/WiderFace-Evaluation
# https://blog.paperspace.com/mean-average-precision/
# https://en.wikipedia.org/wiki/Precision_and_recall


def calc_iou_matrix(a, b):
    """Calculates intersection over union for every pair of two groups of boxes.
    I.e. if a.shape = (n, 4) and b.shape = (m, 4), then result.shape = (n, m),
    and result[i][j] = IoU between box a[i] and box b[j].
    """
    # same as "pairs = np.array(list(itertools.product(a, b))).reshape(-1, 8)" but faster
    idx_a = np.arange(a.shape[0])
    idx_b = np.arange(b.shape[0])
    idx_cartesian = np.array(np.meshgrid(idx_a, idx_b)).T.reshape(-1, 2)
    pairs = np.hstack((a[idx_cartesian[:, 0]], b[idx_cartesian[:, 1]]))

    ix1 = np.maximum(pairs[:, 0], pairs[:, 4])
    iy1 = np.maximum(pairs[:, 1], pairs[:, 5])
    ix2 = np.minimum(pairs[:, 2], pairs[:, 6])
    iy2 = np.minimum(pairs[:, 3], pairs[:, 7])
    iw = np.maximum(0, ix2 - ix1 + 1)
    ih = np.maximum(0, iy2 - iy1 + 1)
    inter = iw * ih
    area1 = (pairs[:, 2] - pairs[:, 0] + 1) * (pairs[:, 3] - pairs[:, 1] + 1)
    area2 = (pairs[:, 6] - pairs[:, 4] + 1) * (pairs[:, 7] - pairs[:, 5] + 1)
    iou = inter / (area1 + area2 - inter)
    iou = iou.reshape(a.shape[0], b.shape[0])
    return iou


def norm_scores(preds):
    """Normalizes prediction scores, stretching them from [min, max] to [0, max] range,
    where "min" and "max" are calculated across the entire validation dataset.
    """
    min_score = min([np.min(p[:, -1]) for p in preds if p.any()])
    max_score = max([np.max(p[:, -1]) for p in preds if p.any()])
    for p in preds:
        p[:, -1] = (p[:, -1] - min_score) / (max_score - min_score)
    return preds


def divide_no_nan(a, b):
    """Divides two numpy arrays as usual except that, at positions where
    the 2nd array's elements are 0, the result is 0 instead of NaN.
    """
    return np.divide(a, b, out=np.zeros(a.shape, dtype=float), where=b!=0)


def calc_pr_curve(pred, gt, iou_thr=0.5, num_score_thr=1000):
    """Calculates a group of precision/recall values for every image by cutting of ``pred``
    at ``num_score_thr`` different score thresholds and then checking how many of the
    remaining predicted boxes has IoU >= ``iou_thr`` with ground truth boxes ``gt``.
    """
    score_thresholds = np.arange(0, 1, 1 / num_score_thr)[::-1]
    total_needed = 0
    det_needed = np.zeros(len(score_thresholds), int)
    det_total = np.zeros(len(score_thresholds), int)
    det_ious = []
    
    print('Comparing predictions with ground truths (IoU threshold = %s)' % str(iou_thr))
    for k in tqdm(range(len(pred))):
        
        total_needed += len(gt[k])
        if (len(pred[k]) == 0 or len(gt[k]) == 0):
            continue
            
        iou = calc_iou_matrix(pred[k][:, :4], gt[k].astype(np.float32))
        mx = iou.max(axis=1)
        mxj = iou.argmax(axis=1)
        pred_n_per_t = [np.count_nonzero(pred[k][:, 4] >= t) for t in score_thresholds]

        mxj[mx < iou_thr] = -1
        idup = np.delete(np.arange(len(mx)), np.unique(mxj, return_index=True)[1])
        mxj[idup] = -1
        found_flags = (mxj >= 0).astype(int)
        found_cumsum = np.concatenate(([0], np.cumsum(found_flags)))
   
        det_needed += found_cumsum[pred_n_per_t]
        det_total += pred_n_per_t
        det_ious.extend(list(mx[found_flags == 1]))

    precision = divide_no_nan(det_needed, det_total)
    recall = divide_no_nan(det_needed, total_needed)

    return precision, recall, score_thresholds, np.mean(det_ious)


def calc_pr_curve_with_settings(pred, gt, st, iou_thr=0.5, num_score_thr=1000):
    """Same as calc_pr_curve but supports different difficulty settings of WIDER
    (or one might also in general call them "subsets" or "filters", and in theory
    use for custom purposes, e.g. adding indices of boxes bigger than certain size)
    """
    num_st = len(st[0])
    score_thresholds = np.arange(0, 1, 1 / num_score_thr)[::-1]
    total_needed = np.zeros((num_st, 1), int)
    det_needed = np.zeros((num_st, len(score_thresholds)), int)
    det_total = np.zeros((num_st, len(score_thresholds)), int)
    det_ious = [[] for _ in range(num_st)]
    
    print('Comparing predictions with ground truths (IoU threshold = %s)' % str(iou_thr))
    for k in tqdm(range(len(pred))):

        iou = calc_iou_matrix(pred[k][:, :4], gt[k].astype(np.float32))
        mx = iou.max(axis=1)
        mxj = iou.argmax(axis=1)
        pred_n_per_t = [np.count_nonzero(pred[k][:, 4] >= t) for t in score_thresholds]
        
        for s, idx in enumerate(st[k]):
            total_needed[s][0] += len(idx)
            found, ignor = 0, 0
            found_cumsum = np.zeros(1 + len(mx), int)
            ignor_cumsum = np.zeros(1 + len(mx), int)
            already = []
            for i in range(len(mx)):
                if mx[i] >= iou_thr and mxj[i] not in already:
                    if mxj[i] in idx:
                        found += 1
                        det_ious[s].append(mx[i])
                    else:
                        ignor += 1
                    already.append(mxj[i])
                found_cumsum[i + 1] = found
                ignor_cumsum[i + 1] = ignor
            det_needed[s] += found_cumsum[pred_n_per_t]
            det_total[s] += pred_n_per_t - ignor_cumsum[pred_n_per_t]

    precision = divide_no_nan(det_needed, det_total)
    recall = divide_no_nan(det_needed, total_needed)
    avg_iou = [np.mean(det_ious[i]) for i in range(num_st)]
    return precision, recall, score_thresholds, avg_iou


def calc_ap(precision, recall):
    """Computes average precision (AP) for a precision-recall curve as described here:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
    
    Technically, one could instead calculate area under curve with np.trapz or sklearn.metrics.auc,
    but the link above states that for pr-curves the weighted mean is considered superior.

    Also ensures that the curve is monotonically decreasing first by "elevating" the dipping parts,
    so [0.7, 0.3, 0.4, 0.5] becomes [0.7, 0.5, 0.5, 0.5] (but [0.7, 0.6, 0.5, 0.4] stay the same)
    (implemented by going backwards and replacing every element with the current maximum:
    https://stackoverflow.com/a/28563925).

    :param precision: 1D np.array of precisions, representing Y coordinates of dots composing a curve
    :param recall: 1D np.array of recalls, representing X coodrinates of dots composing a curve
    :return: Average precision, a single float value
    """
    R = np.concatenate(([0.], recall))
    P = np.concatenate(([0.], precision))
    P = np.maximum.accumulate(P[::-1])[::-1]
    ap = np.sum((R[1:] - R[:-1]) * P[1:])
    return ap


def best_f1(precision, recall, score_thrs):
    """Returns best F1 score from a range of precisions/recalls,
    along with the score threshold where it occured.
    """
    F1 = divide_no_nan(2 * precision * recall, precision + recall)
    return (np.max(F1), score_thrs[np.argmax(F1)])