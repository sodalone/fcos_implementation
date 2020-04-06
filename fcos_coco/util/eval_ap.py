# -*- coding: utf-8 -*-
# @Author : soda
#
# @License : Copyright © 2020 soda <soda791401863@gmail.com>
#
# @Contact : soda791401863@gmail.com
#
# @TIME : 2020-03-22 23:02:59
'''

'''

import numpy as np
from collections import defaultdict

def pred_target_iou(pred_box, target_box, eps=1e-6):
    xy_min = np.maximum(pred_box[:, None, :2], target_box[:, :2])
    xy_max = np.minimum(pred_box[:, None, 2:], target_box[:, 2:])
    overlap = np.prod(xy_max - xy_min, axis=2) * (xy_max > xy_min).all(axis=2)
    pred_area = np.prod(pred_box[:, :2] - pred_box[:, 2:], axis=1)
    target_area = np.prod(target_box[:, :2] - target_box[:, 2:], axis=1)
    return overlap / (pred_area[:, None] + target_area - overlap + eps)

def get_prec_recall(pred_boxes, pred_labels, pred_scores,
                    target_boxes, target_labels, iou_th=0.5):
    n_num = defaultdict(int)
    score = defaultdict(list)
    mark = defaultdict(list)

    for i in range(len(pred_boxes)):
        pred_box = pred_boxes[i]
        pred_label = pred_labels[i]
        pred_score = pred_scores[i]
        target_box = target_boxes[i]
        target_label = target_labels[i]

        for l in np.unique(np.concatenate((pred_label, target_label)).astype(int)):
            pred_map = pred_label == l
            pred_box_l = pred_box[pred_map]
            pred_score_l = pred_score[pred_map]
            pred_sort = pred_score_l.argsort()[::-1]
            pred_box_l = pred_box_l[pred_sort]
            pred_score_l = pred_score_l[pred_sort]

            target_map = target_label == l
            target_box_l = target_box[target_map]

            n_num[l] += target_map.sum()
            score[l].extend(pred_score_l)

            if len(pred_box_l) == 0:
                continue
            if len(target_box_l) == 0:
                mark[l].extend((0,) * pred_box_l.shape[0])
                continue
            
            pred_box_l = pred_box_l.copy()
            pred_box_l[:, 2:] += 1
            target_box_l = target_box_l.copy()
            target_box_l[:, 2:] += 1

            iou = pred_target_iou(pred_box_l, target_box_l)
            index = iou.argmax(axis=1)
            index[iou.max(axis=1) < iou_th] = -1
            del iou

            flag = np.zeros(target_box_l.shape[0], dtype=bool)

            for idx in index:
                if idx >= 0:
                    if flag[idx]:
                        mark[l].append(0)
                    else:
                        mark[l].append(1)
                    flag[idx] = True
                else:
                    mark[l].append(0)
    
    class_num = max(n_num.keys()) + 1

    prec = [None] * class_num
    recall = [None] * class_num

    for l in n_num.keys():
        score_l = np.array(score[l])
        mark_l = np.array(mark[l])
        sort = score_l.argsort()[::-1]
        mark_l = mark_l[sort]

        tp = np.cumsum(mark_l == 1)
        fp = np.cumsum(mark_l == 0)

        prec[l] = tp / (tp + fp)
        if n_num[l] > 0:
            recall[l] = tp / n_num[l]
    
    return prec, recall


def get_ap(prec, recall):
    class_num = len(prec)
    ap = np.empty(class_num)
    for i in range(class_num):
        if prec[i] is None or recall[i] is None:
            ap[i] = np.nan
            continue

        prec_i = np.concatenate(([0], np.nan_to_num(prec[i]), [0]))
        prec_i = np.maximum.accumulate(prec_i[::-1])[::-1]

        recall_i = np.concatenate(([0], recall[i], [1]))
        
        mark = np.where(recall_i[1:] != recall_i[:-1])[0]

        ap[i] = np.sum((recall_i[mark + 1] - recall_i[mark]) * prec_i[mark + 1])
    
    return ap

def eval_ap(pred_boxes, pred_labels, pred_scores, target_boxes, target_labels, iou_th=0.5):
    prec, recall = get_prec_recall(pred_boxes, pred_labels, pred_scores,
                                   target_boxes, target_labels, iou_th)
    ap = get_ap(prec, recall)
    
    return {
        'ap': ap,
        'map': np.nanmean(ap)
    }
