# -*- coding: utf-8 -*-
# @Author : soda
#
# @License : Copyright © 2020 soda <soda791401863@gmail.com>
#
# @Contact : soda791401863@gmail.com
#
# @TIME : 2020-03-20 14:06:47
'''
fcos主网络
'''
import math
import numpy as np
from time import time
from util.assign_box import assign_box_cupy, assign_box
from util.focal_loss import focal_loss
from util.iou_loss import iou_loss
from util.BCE_loss import BCE_loss
from util.nms import nms
from model.ResNet50 import resnet50 as backbone
import torch.nn.functional as F
from torch import nn
import torch


class Fcos(nn.Module):
    def __init__(self, size, classes, pic, pretrained=False):
        super().__init__()

        self.size = size
        self.classes = classes
        self.max_detections = 3000
        self.ltrb_limit = [[-1, 64], [64, 128],
                           [128, 256], [256, 512], [512, 1024]]
        self.pic = pic
        self.r = [12, 24, 48, 96, 192]

        self.backbone = backbone(pretrained=pretrained)
        self.relu = nn.ReLU(inplace=True)
        self.prj_5 = nn.Conv2d(2048, 256, kernel_size=1)
        self.prj_4 = nn.Conv2d(1024, 256, kernel_size=1)
        self.prj_3 = nn.Conv2d(512, 256, kernel_size=1)
        self.conv_5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv_6 = nn.Conv2d(2048, 256, kernel_size=3, stride=2, padding=1)
        self.conv_7 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)

        self.conv_cls_center = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True))
        
        self.conv_cls = nn.Conv2d(256, self.classes, kernel_size=3, padding=1)
        self.conv_center = nn.Conv2d(256, 1, kernel_size=3, padding=1)

        self.conv_reg = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 4, kernel_size=3, padding=1))

        init_layers = [self.conv_cls_center, self.conv_cls, self.conv_center, self.conv_reg]
        for init_layer in init_layers:
            for item in init_layer.children():
                if isinstance(item, nn.Conv2d) or isinstance(item, nn.ConvTranspose2d):
                    nn.init.constant_(item.bias, 0)
                    nn.init.normal_(item.weight, mean=0, std=0.01)

        pi = 0.01
        _bias = -math.log((1.0-pi)/pi)  # retinanet bias初始化
        nn.init.constant_(self.conv_cls.bias, _bias)

        self.scale_param = nn.Parameter(
            torch.FloatTensor([32, 64, 128, 256, 512]))
        self.scale_param.requires_grad = False

    def upsample(self, m):
        return F.interpolate(m, size=(m.shape[2]*2-1, m.shape[3]*2-1),
                             mode='bilinear', align_corners=True)

    def decode_box(self, box, size, pic):
        #  [offset_y, offset_x, h, w] -> [ymin, xmin, ymax, xmax]

        x_y = torch.linspace(0, size-1, pic).to(box.device)
        # 生成坐标图
        center_y, center_x = torch.meshgrid(x_y, x_y)
        center_y = center_y.squeeze(0)
        center_x = center_x.squeeze(0)

        cx = box[:, :, :, 0] + center_x
        cy = box[:, :, :, 1] + center_y
        cw = box[:, :, :, 2]
        ch = box[:, :, :, 3]

        xmin = cx - cw/2.0
        ymin = cy - ch/2.0
        xmax = cx + cw/2.0
        ymax = cy + ch/2.0
        return torch.stack([xmin, ymin, xmax, ymax], dim=3)

    def test_decode(self, pred_cls, pred_center, pred_reg, local):
        pred_cls_s, pred_cls_i = torch.max(pred_cls.sigmoid(), dim=2)
        pred_cls_s = pred_cls_s * pred_center.sigmoid()
        pred_cls_i = pred_cls_i + 1

        num_max = min(self.max_detections, pred_cls_i.shape[1])
        top = torch.topk(pred_cls_s, num_max, largest=True, dim=1)[1]
        cls_s, cls_i, reg_i = [], [], []
        for b in range(pred_cls.shape[0]):
            cls_s.append(pred_cls_s[b][top[b]])
            cls_i.append(pred_cls_i[b][top[b]])
            pred_reg_b = pred_reg[b][top[b]]
            pred_reg_b[:, 0].clamp_(min=float(local[b, 0]))
            pred_reg_b[:, 1].clamp_(min=float(local[b, 1]))
            pred_reg_b[:, 2].clamp_(max=float(local[b, 2]))
            pred_reg_b[:, 3].clamp_(max=float(local[b, 3]))
            reg_i.append(pred_reg_b)
        return torch.stack(cls_i), torch.stack(cls_s), torch.stack(reg_i)

    def forward(self, img, local, label_class=None, label_box=None):
        C3, C4, C5 = self.backbone(img)
        P5 = self.prj_5(C5)
        P4 = self.prj_4(C4)
        P3 = self.prj_3(C3)
        P4 = P4 + self.upsample(P5)
        P3 = P3 + self.upsample(P4)
        P3 = self.conv_3(P3)
        P4 = self.conv_4(P4)
        P5 = self.conv_5(P5)
        P6 = self.conv_6(C5)
        P7 = self.conv_7(self.relu(P6))

        pred_cls = []
        pred_center = []
        pred_reg = []
        for i, feature in enumerate([P3, P4, P5, P6, P7]):
            cls_center_i = self.conv_cls_center(feature)
            cls_i = self.conv_cls(cls_center_i)
            center_i = self.conv_center(cls_center_i)
            reg_i = self.conv_reg(feature) * self.scale_param[i]

            cls_i = cls_i.permute(0, 2, 3, 1).contiguous()
            center_i = center_i.permute(0, 2, 3, 1).contiguous()
            reg_i = reg_i.permute(0, 2, 3, 1).contiguous()

            reg_i = self.decode_box(reg_i, self.size, self.pic[i])
            pred_cls.append(cls_i.view(cls_i.shape[0], -1, self.classes))
            pred_center.append(center_i.view(center_i.shape[0], -1))
            pred_reg.append(reg_i.view(reg_i.shape[0], -1, 4))

        # 共享特征头
        pred_cls = torch.cat(pred_cls, dim=1)
        pred_center = torch.cat(pred_center, dim=1)
        pred_reg = torch.cat(pred_reg, dim=1)

        # train
        if label_class is not None and label_box is not None:
            n_max = min(label_class.shape[-1], 200)
            if n_max == 200:
                label_class = label_class[:, :200]
                label_box = label_box[:, :200, :]

            target_cls = []
            target_center = []
            target_reg = []
            for i in range(len(self.pic)):
                target_cls_i, target_center_i, target_reg_i = assign_box_cupy(label_class, label_box, local,
                                                        self.size, self.pic[i], self.ltrb_limit[i][0],
                                                        self.ltrb_limit[i][1], self.r[i])
                target_cls.append(target_cls_i.view(target_cls_i.shape[0], -1))
                target_center.append(target_center_i.view(target_center_i.shape[0], -1))
                target_reg.append(target_reg_i.view(target_reg_i.shape[0], -1, 4))
            target_cls = torch.cat(target_cls, dim=1)
            target_center = torch.cat(target_center, dim=1)
            target_reg = torch.cat(target_reg, dim=1)

            map_cls = target_cls > -1
            map_reg = target_cls > 0
            num_map = torch.sum(map_reg, dim=1).clamp(min=1)
            loss = []

            for b in range(label_class.shape[0]):
                pred_cls_map = pred_cls[b][map_cls[b]]
                target_cls_map = target_cls[b][map_cls[b]]
                
                pred_center_map = pred_center[b][map_reg[b]]
                target_center_map = target_center[b][map_reg[b]]

                pred_reg_map = pred_reg[b][map_reg[b]]
                target_reg_map = target_reg[b][map_reg[b]]

                loss_cls = focal_loss(pred_cls_map, target_cls_map).sum().view(1)
                loss_center = BCE_loss(pred_center_map, target_center_map).sum().view(1)
                loss_reg = iou_loss(pred_reg_map, target_reg_map).sum().view(1)
                loss.append((loss_cls + loss_reg + loss_center) / float(num_map[b]))
            loss = torch.cat(loss, dim=0)
            return loss

        else:
            return self.test_decode(pred_cls, pred_center, pred_reg, local)

def get_pred(temp, nms_th, nms_iou):
    cls_i, cls_s, reg_i = temp
    pred_cls_i, pred_cls_s, pred_reg_i = [], [], []
    for b in range(cls_i.shape[0]):
        map_th = cls_s[b] > nms_th
        pred_cls_i_b = cls_i[b][map_th]
        pred_cls_s_b = cls_s[b][map_th]
        pred_reg_i_b = reg_i[b][map_th]
        map_nms = nms(pred_reg_i_b, pred_cls_s_b, nms_iou)
        pred_cls_i.append(pred_cls_i_b[map_nms])
        pred_cls_s.append(pred_cls_s_b[map_nms])
        pred_reg_i.append(pred_reg_i_b[map_nms])
    return pred_cls_i, pred_cls_s, pred_reg_i

