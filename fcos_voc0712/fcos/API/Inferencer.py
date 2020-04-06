# -*- coding: utf-8 -*-
# @Author : soda
#
# @License : Copyright © 2020 soda <soda791401863@gmail.com>
#
# @Contact : soda791401863@gmail.com
#
# @TIME : 2020-03-25 17:17:41
'''

'''
import torch
import torchvision
from PIL import Image
from model.fcos import get_pred
from dataset.dataset import center_fix

class Inferencer(object):
    def __init__(self, net):
        self.net = net
        self.normalizer = torchvision.transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))

    def pred(self, img, size, nms_th, nms_iou):
        temp_box = torch.zeros(0, 4)
        img, box, local, scale = center_fix(img, temp_box, size)
        img = torchvision.transforms.ToTensor()(img)
        img = self.normalizer(img).view(1, img.shape[0], img.shape[1], img.shape[2])
        img = img.cuda()
        local = local.view(1, -1).cuda()
        with torch.no_grad():
            pred_cls_i, pred_cls_s, pred_reg_i = get_pred(self.net(img, local), nms_th, nms_iou)
            pred_reg_i[0][:, 0::2] -= local[0, 0]
            pred_reg_i[0][:, 1::2] -= local[0, 1]
            pred_reg_i[0] /= scale
        return pred_cls_i[0], pred_cls_s[0], pred_reg_i[0]