from .loader import prepare_dataset
from .parser import get_set_data, get_wider_data
from .predictor import get_predictions
from .calculator import calc_pr_curve, calc_ap, best_f1, norm_scores, calc_pr_curve_with_settings


def eval_det(set_name, model, load=None, pad_mult=32, batch_size=32, iou_threshold=0.5):
    check_set_name(set_name)
    updir, imdir, gtdir = prepare_dataset(set_name)
    fn, gt = get_set_data(set_name, gtdir)
    pred = get_predictions(load, model, fn, updir, imdir, pad_mult, batch_size)
    precision, recall, score_thrs, avg_iou = calc_pr_curve(pred, gt, iou_threshold, num_score_thr=100)
    ap = calc_ap(precision, recall)
    f1 = best_f1(precision, recall, score_thrs)
    print("AP: %.16f" % ap)
    print("Mean IoU: %.3f" % avg_iou)
    print("Best F1 score: %.3f (for score threshold = %.3f)" % f1)


def eval_det_wider(model, load=None, pad_mult=32, batch_size=32, iou_threshold=0.5):
    updir, imdir, gtdir = prepare_dataset('WIDER_FACE')
    fn, gt, st = get_wider_data(gtdir)
    pred = get_predictions(load, model, fn, updir, imdir, pad_mult, batch_size)
    pred = norm_scores(pred)
    precision, recall, score_thrs, avg_iou = calc_pr_curve_with_settings(pred, gt, st, iou_threshold)
    ap = [calc_ap(precision[i], recall[i]) for i in range(3)]
    f1 = [best_f1(precision[i], recall[i], score_thrs) for i in range(3)]
    sett = ['Easy  ', 'Medium', 'Hard  ']
    for i in range(3):
        print("%s AP: %.16f" % (sett[i], ap[i]))
    for i in range(3):
        print("%s mean IoU: %.3f" % (sett[i], avg_iou[i]))
    for i in range(3):
        print("%s best F1 score: %.3f (for score threshold = %.3f)" % (sett[i], f1[i][0], f1[i][1]))


def check_set_name(nm):
    sets = ['FDDB', 'ICARTOON', 'PIXIV2018', 'PIXIV2018_ORIG']
    sets_txt = ', '.join(['"%s"' % s for s in sets])
    if nm not in sets:
        raise ValueError('Unknown set_name. Possible values are %s' % sets_txt)