from functools import partial

import torch
import torch.nn.functional as F

from .bbox import calc_iou_matrix, convert_to_xyxy, convert_to_cwh, encode_boxes


def assign(gtboxes, priors, low_thr, high_thr, match_low_quality):
    """"""
    res = []
    for i in range(len(gtboxes)):
        pr = priors[i] if isinstance(priors, list) else priors
        m = calc_iou_matrix(gtboxes[i], pr)
        v, idx = m.max(dim=0)
        if match_low_quality:
            copied = idx.clone()
        idx[v < low_thr] = -1
        idx[(v >= low_thr) & (v < high_thr)] = -2
        if match_low_quality:
            maxgt = m.max(dim=1)[0]
            extra = torch.where(m == maxgt[:, None])[1]
            idx[extra] = copied[extra]
        res.append(idx)
    return res


def sample_random(gtidxs, num, pos_fraction):
    sidx_all, sidx_pos = [], []
    for gtidx in gtidxs:
        pos = torch.nonzero(gtidx >= 0).squeeze()
        neg = torch.nonzero(gtidx == -1).squeeze()
        np = min(pos.numel(), int(num * pos_fraction))
        nn = min(neg.numel(), num - np)
        perm1 = torch.randperm(pos.numel())[:np]
        perm2 = torch.randperm(neg.numel())[:nn]
        pos = pos[perm1]
        neg = neg[perm2]
        sidx_all.append(torch.cat([pos, neg]))
        sidx_pos.append(pos)
    return sidx_all, sidx_pos


def get_matched_targets(gtboxes, priors, gtidxs, sidx_pos, enc_weights=(1, 1)):
    targets = []
    for i in range(len(gtboxes)):
        pr = priors[i] if isinstance(priors, list) else priors
        pos = sidx_pos[i]
        t = gtboxes[i][gtidxs[i][pos]]
        t = encode_boxes(t, pr[pos], enc_weights)
        targets.append(t)
    return targets
        
        
def get_matched_labels(gtlabels, gtidxs, sidx_all):
    labels = []
    for i in range(len(gtidxs)):
        all_ = sidx_all[i]
        if gtlabels is None:
            lb = gtidxs[i][all_].clamp(max=0) + 1
        else:
            lb = gtidxs[i][all_].clone()
            lb[lb >= 0] = gtlabels[i][lb[lb >= 0]]
            lb[lb == -1] = 0
        labels.append(lb)
    return labels


def binary_cross_entropy(inputs, targets, reduction):
    targets = targets.to(inputs.dtype)
    return F.binary_cross_entropy_with_logits(inputs, targets, reduction=reduction)


def sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2, reduction='sum'):
    # https://github.com/pytorch/vision/blob/main/torchvision/ops/focal_loss.py
    
    expanded = torch.zeros((targets.shape[0], 91), dtype=torch.float32)
    expanded[targets > 0, targets[targets > 0]] = 1.0
    targets = expanded

    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.sum()


loss_funcs = {
    'ce_bin': binary_cross_entropy,
    'ce': F.cross_entropy,
    'focal': sigmoid_focal_loss,
    'l1': F.l1_loss,
    'l1_s': partial(F.smooth_l1_loss, beta=1/9)
}


def calc_losses(types, logs, regs, labels, targets, avg_mode='total', avg_divs='usual'):
    assert avg_mode in ['total', 'per_image']
    assert avg_divs in ['usual', 'always_pos', 'always_all']
    cls_func = loss_funcs[types[0]]
    reg_func = loss_funcs[types[1]]
    if avg_mode == 'total':
        logs, regs, labels, targets = [torch.cat(x) for x in [logs, regs, labels, targets]]
        div1 = labels.shape[0] if avg_divs != 'always_pos' else targets.shape[0]
        div2 = targets.shape[0] if avg_divs != 'always_all' else labels.shape[0]
        cls_loss = cls_func(logs, labels, reduction='sum') / div1
        reg_loss = reg_func(regs, targets, reduction='sum') / div2
    else:
        cls_loss, reg_loss = [], []
        n = len(logs)
        for i in range(n):
            div1 = labels[i].shape[0] if avg_divs != 'always_pos' else targets[i].shape[0]
            div2 = targets[i].shape[0] if avg_divs != 'always_all' else labels[i].shape[0]
            cls_loss.append(cls_func(logs[i], labels[i], reduction='sum') / div1)
            reg_loss.append(reg_func(regs[i], targets[i], reduction='sum') / div2)
        cls_loss = sum(cls_loss) / n
        reg_loss = sum(reg_loss) / n
    return cls_loss, reg_loss


def get_losses(gtboxes, gtlabels, priors, regs, logs, matcher, sampler, types, avg_mode='total', avg_divs='usual'):
    """"""
    gtidxs = assign(gtboxes, convert_to_xyxy(priors), *matcher)
    if sampler is not None:
        sidx_all, sidx_pos = sample_random(gtidxs, *sampler)
    else:
        sidx_all = [torch.nonzero(gtidx != -2).squeeze() for gtidx in gtidxs]
        sidx_pos = [torch.nonzero(gtidx >= 0).squeeze() for gtidx in gtidxs]    
    
    targets = get_matched_targets(gtboxes, priors, gtidxs, sidx_pos)
    labels = get_matched_labels(gtlabels, gtidxs, sidx_all)
    
    n = len(gtboxes)
    logs = [logs[i][sidx_all[i]].squeeze() for i in range(n)]
    regs = [regs[i][sidx_pos[i]] for i in range(n)]
    res = calc_losses(types, logs, regs, labels, targets, avg_mode, avg_divs)
    return res


def match_with_targets(gtboxes, gtlabels, proposals, bg_iou, fg_iou, match_lq, batch, pos_ratio):
    mt, ml, sa, sp = [], [], [], []
    gtidxs = assign(gtboxes, proposals, bg_iou, fg_iou, match_lq)
    sidx_all, sidx_pos = sample_random(gtidxs, batch, pos_ratio)
    proposals = [convert_to_cwh(p) for p in proposals]
    targets = get_matched_targets(gtboxes, proposals, gtidxs, sidx_pos, (0.1, 0.2))
    labels = get_matched_labels(gtlabels, gtidxs, sidx_all)
    return targets, labels, sidx_all