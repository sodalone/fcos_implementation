# -*- coding: utf-8 -*-
# @Author : soda
#
# @License : Copyright © 2020 soda <soda791401863@gmail.com>
#
# @Contact : soda791401863@gmail.com
#
# @TIME : 2020-03-23 20:51:47
'''

'''

import torch
from time import time
import numpy as np
from tqdm import tqdm

from util.eval_ap import eval_ap
from model.fcos import get_pred

class Tester(object):
    def __init__(self, net, test_data, test_loader, nms_th, nms_iou):
        self.net = net
        self.test_data = test_data
        self.test_loader = test_loader
        self.nms_th = nms_th
        self.nms_iou = nms_iou

    def start(self):
        with torch.no_grad():
            self.net.eval()
            pred_boxes, pred_labels, pred_scores, target_boxes, target_labels = [], [], [], [], []
            all_batch_size = 0
            for imgs, boxes, labels, locals, scales in self.test_loader:
                batch_size = imgs.shape[0]
                all_batch_size += batch_size
                target_boxes_i, target_labels_i = [], []
                for b in range(batch_size):
                    map = labels[b] > 0
                    target_boxes_i.append(boxes[b][map].detach().cpu().numpy())
                    target_labels_i.append(labels[b][map].detach().cpu().numpy())

                pred_cls_i, pred_cls_s, pred_reg_i = get_pred(self.net(imgs, locals),
                                                              self.nms_th, self.nms_iou)
                for b in range(len(pred_cls_i)):
                    pred_cls_i[b] = pred_cls_i[b].detach().cpu().numpy()
                    pred_cls_s[b] = pred_cls_s[b].detach().cpu().numpy()
                    pred_reg_i[b] = pred_reg_i[b].detach().cpu().numpy()

                pred_boxes = pred_boxes + pred_reg_i
                pred_labels = pred_labels + pred_cls_i
                pred_scores = pred_scores + pred_cls_s
                target_boxes = target_boxes + target_boxes_i
                target_labels = target_labels + target_labels_i
                print('test: {}/{}'.format(all_batch_size, len(self.test_data)), end='\r')
                
            ap_iou = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            ap = []
            for iou_th in ap_iou:
                result = eval_ap(pred_boxes, pred_labels, pred_scores,
                                 target_boxes, target_labels, iou_th=iou_th)
                ap.append(result)
            ap_sum = 0.0
            for i in range(len(ap)):
                ap_sum += ap[i]['map']

            map_mean = ap_sum / len(ap)
            map_50 = ap[0]['map']
            map_75 = ap[5]['map']
            print('map_mean: %f, map_50: %f, map_75: %f' %
                  (map_mean, map_50, map_75))
            self.net.train()
            return map_mean, map_50, map_75
