# -*- coding: utf-8 -*-
# @Author : soda
#
# @License : Copyright © 2020 soda <soda791401863@gmail.com>
#
# @Contact : soda791401863@gmail.com
#
# @TIME : 2020-03-29 17:02:19
'''

'''

import torch
import json
from model.fcos import get_pred
from tqdm import tqdm

class COCO_eval(object):

    def __init__(self, net, test_data, val_image_ids, coco_labels):
        self.net = net
        self.test_data = test_data
        self.val_image_ids = val_image_ids
        self.coco_labels = coco_labels

    def start(self):
        with torch.no_grad():
            results = []
            num = tqdm(range(len(self.test_data)))
            for i in num:
                img, bbox, label, loc, scale = self.test_data[i]
                img = img.cuda().view(1, img.shape[0], img.shape[1], img.shape[2])
                loc = loc.cuda().view(1, -1)
                pred_cls_i, pred_cls_s, pred_reg_i = get_pred(self.net(img, loc),
                                                            self.net.nms_th, self.net.nms_iou)
                pred_cls_i = pred_cls_i[0].cpu()
                pred_cls_s = pred_cls_s[0].cpu()
                pred_reg_i = pred_reg_i[0].cpu()
                if pred_reg_i.shape[0] > 0:
                    xmin, ymin, xmax, ymax = pred_reg_i.split(
                        [1, 1, 1, 1], dim=1)
                    h, w = ymax - ymin, xmax - xmin
                    pred_reg_i = torch.cat(
                        [xmin - loc[0, 0].cpu(), ymin - loc[0, 1].cpu(), w, h], dim=1)
                    pred_reg_i = pred_reg_i / float(scale)

                    for box_id in range(pred_reg_i.shape[0]):
                        score = float(pred_cls_s[box_id])
                        label = int(pred_cls_i[box_id])
                        box = pred_reg_i[box_id, :]
                        image_result = {
                            'image_id': self.val_image_ids[i],
                            'category_id': self.coco_labels[str(label)],
                            'score': float(score),
                            'bbox': box.tolist(),
                        }
                        results.append(image_result)
                num.set_description()
            json.dump(results, open('coco_bbox_results.json', 'w'), indent=4)
