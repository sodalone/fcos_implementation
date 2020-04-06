# -*- coding: utf-8 -*-
# @Author : soda
#
# @License : Copyright © 2020 soda <soda791401863@gmail.com>
#
# @Contact : soda791401863@gmail.com
#
# @TIME : 2020-03-21 20:24:42
'''

'''

import torch
import numpy as np

def nms(boxes, scores, th):

    if boxes.shape[0] == 0:
        return torch.zeros(0).long()
 
    boxes = boxes.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy()
    eps = 1e-10
    xmin, ymin, xmax, ymax = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (xmax - xmin + eps) * (ymax - ymin + eps)
    
    map_nms = []
    index = scores.argsort()[::-1]
    while index.size > 0:
        i = index[0]
        map_nms.append(i)

        xmin_all = np.maximum(xmin[i], xmin[index[1:]])
        ymin_all = np.maximum(ymin[i], ymin[index[1:]])
        xmax_all = np.minimum(xmax[i], xmax[index[1:]])
        ymax_all = np.minimum(ymax[i], ymax[index[1:]])

        w = np.maximum(0, xmax_all-xmin_all+eps)
        h = np.maximum(0, ymax_all-ymin_all+eps)

        overlaps = w * h

        ious = overlaps / (areas[i]+areas[index[1:]] - overlaps)

        idx = np.where(ious <= th)[0]

        index = index[idx+1]

    return torch.LongTensor(map_nms)