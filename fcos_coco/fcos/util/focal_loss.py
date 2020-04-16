# -*- coding: utf-8 -*-
# @Author : soda
#
# @License : Copyright © 2020 soda <soda791401863@gmail.com>
#
# @Contact : soda791401863@gmail.com
#
# @TIME : 2020-03-21 11:33:15
'''

'''

import torch
from torch.nn import functional as F


def focal_loss(pred, target, gamma=2.0, alpha=0.25):
    one_hot = torch.zeros(pred.shape[0], 
            1 + pred.shape[1]).to(pred.device).scatter_(1, 
                target.view(-1,1), 1)
    one_hot = one_hot[:, 1:]
    pred = pred.sigmoid()
    ce_loss = -1 * (pred.log()*one_hot + (1.0-pred).log()*(1.0-one_hot))
    pt = pred*one_hot + (1.0-pred)*(1.0-one_hot)
    w = alpha*one_hot + (1.0-alpha)*(1.0-one_hot)
    w = w * torch.pow((1.0-pt), gamma)
    loss = w * ce_loss
    return loss