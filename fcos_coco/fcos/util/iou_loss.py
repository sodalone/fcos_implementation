# -*- coding: utf-8 -*-
# @Author : soda
#
# @License : Copyright © 2020 soda <soda791401863@gmail.com>
#
# @Contact : soda791401863@gmail.com
#
# @TIME : 2020-03-21 11:33:08
'''

'''

import torch
import numpy as np
import cupy as cp
from cupy.core.dlpack import toDlpack
from cupy.core.dlpack import fromDlpack
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack


def iou_loss(pred, target, eps=1e-6):
    xy_min = torch.max(pred[:, :2], target[:, :2])
    xy_max = torch.min(pred[:, 2:], target[:, 2:])
    hw = (xy_max - xy_min + 1).clamp(min=0)
    overlap = hw[:, 0] * hw[:, 1]

    pred_area = (pred[:, 2] - pred[:, 0] + 1) * (pred[:, 3] - pred[:, 1] + 1)
    target_area = (target[:, 2] - target[:, 0] + 1) * (target[:, 3] - target[:, 1] + 1)
    iou = overlap / (pred_area + target_area - overlap)
    iou = iou.clamp(min=eps) 
    
    loss = -iou.log()
    return loss