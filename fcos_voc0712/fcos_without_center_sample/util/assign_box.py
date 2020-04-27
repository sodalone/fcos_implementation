# -*- coding: utf-8 -*-
# @Author : soda
#
# @License : Copyright © 2020 soda <soda791401863@gmail.com>
#
# @Contact : soda791401863@gmail.com
#
# @TIME : 2020-03-20 22:39:17
'''

'''

import torch
import numpy as np
import cupy as cp
from torch.utils.dlpack import from_dlpack, to_dlpack
from cupy.core.dlpack import fromDlpack, toDlpack


def assign_box_cupy(label_cls, label_box, locals, size, pic, ltrb_min, ltrb_max, r):

    cp_out = cp.ElementwiseKernel(
        'raw T label_cls, raw T label_box, raw T locals, \
        int32 pic, int32 size, int32 ltrb_min, int32 ltrb_max, int32 r, \
        int32 scale, int32 batch_size, int32 n_max',
        'T cls, T xmin, T ymin, T xmax, T ymax, T center',
        '''
        int b = i / (pic*pic);
        int j = (i % (pic*pic)) / pic;
        int k = i % pic;

        int x = k * scale;
        int y = j * scale;
        if (x >= locals[4*b+0] && x <= locals[4*b+2] && y >=locals[4*b+1] && y <= locals[4*b+3]){
            float pred_area = size*size;
            float pred_cls, pred_xmin, pred_ymin, pred_xmax, pred_ymax;
            pred_cls = pred_xmin = pred_ymin = pred_xmax = pred_ymax = 0;
            float pred_left, pred_top, pred_right, pred_bottom;
            int n = 0;
            for (n = 0; n < n_max; n++){
                float xmin_t = label_box[b*n_max*4+n*4+0];
                float ymin_t = label_box[b*n_max*4+n*4+1];
                float xmax_t = label_box[b*n_max*4+n*4+2];
                float ymax_t = label_box[b*n_max*4+n*4+3];
                float left = x - xmin_t;
                float top = y - ymin_t;
                float right = xmax_t - x;
                float bottom = ymax_t - y;
                float max_ltrb = max(left, max(top, max(right, bottom)));
                float box_area = (xmax_t - xmin_t) * (ymax_t - ymin_t);
                if (left > 0 && top > 0 && right > 0 && bottom > 0 && max_ltrb > ltrb_min &&
                    max_ltrb < ltrb_max && box_area <= pred_area){
                        pred_area = box_area;
                        pred_cls = label_cls[b*n_max+n];
                        pred_xmin = xmin_t;
                        pred_ymin = ymin_t;
                        pred_xmax = xmax_t;
                        pred_ymax = ymax_t;
                        pred_left = left;
                        pred_top = top;
                        pred_right = right;
                        pred_bottom = bottom;
                }
            }
            if (pred_cls > 0){
                cls = pred_cls;
                xmin = pred_xmin;
                ymin = pred_ymin;
                xmax = pred_xmax;
                ymax = pred_ymax;
                center = sqrt((min(pred_left, pred_right) / max(pred_left, pred_right)) * 
                                (min(pred_top, pred_bottom) / max(pred_top, pred_bottom)));
            }
            else{
                cls = xmin = ymin = xmax = ymax = center = 0;
            }
        }
        else{
            cls = -1;
            xmin = ymin = xmax = ymax = center = 0;
        }
        ''',
        'assign_box'
    )
    label_cls = fromDlpack(to_dlpack(label_cls.float()))
    label_box = fromDlpack(to_dlpack(label_box))
    locals = fromDlpack(to_dlpack(locals))
    scale = int((size - 1) / (pic - 1))
    batch_size, n_max = label_cls.shape
    cls = cp.zeros((batch_size, pic, pic), dtype=cp.float32)
    xmin = cp.zeros((batch_size, pic, pic), dtype=cp.float32)
    ymin = cp.zeros((batch_size, pic, pic), dtype=cp.float32)
    xmax = cp.zeros((batch_size, pic, pic), dtype=cp.float32)
    ymax = cp.zeros((batch_size, pic, pic), dtype=cp.float32)
    center = cp.zeros((batch_size, pic, pic), dtype=cp.float32)
    cp_out(label_cls, label_box, locals, pic, size, ltrb_min, ltrb_max, r,
           scale, batch_size, n_max, cls, xmin, ymin, xmax, ymax, center)
    cls_out = from_dlpack(toDlpack(cls)).long()
    center_out = from_dlpack(toDlpack(center)).float()
    box_out = cp.stack([xmin, ymin, xmax, ymax], axis=3)
    box_out = from_dlpack(toDlpack(box_out)).float()
    return cls_out, center_out, box_out


def assign_box(label_cls, label_box, locals, size, pic, ltrb_min, ltrb_max, r):
    device = label_cls.device
    label_cls = label_cls.detach().cpu().numpy()
    label_box = label_box.detach().cpu().numpy()
    locals = locals.detach().cpu().numpy()
    scale = int((size - 1) / (pic - 1))
    batch_size, n_max = label_cls.shape
    cls_out = np.zeros((batch_size, pic, pic))
    box_out = np.zeros((batch_size, pic, pic, 4))

    for b in range(batch_size):
        for i in range(pic):
            for j in range(pic):
                x = i * scale
                y = j * scale
                if (x >= locals[b][0] and x <= locals[b][2] and
                        y >= locals[b][1] and y <= locals[b][3]):
                    pred_area = float('inf')
                    pred_cls = 0
                    pred_box = np.zeros(4)
                    for n in range(n_max):
                        xmin, ymin, xmax, ymax = label_box[b][n]
                        top, bottom, left, right = y-ymin, ymax-y, x-xmin, xmax-x
                        max_ltrb = max(top, max(bottom, max(left, right)))
                        if (top > 0 and bottom > 0 and left > 0 and right > 0 and
                                max_ltrb > ltrb_min and max_ltrb < ltrb_max):
                            box_area = (ymax - ymin) * (xmax - xmin)
                            if box_area <= pred_area:
                                pred_area = box_area
                                pred_cls = label_cls[b][n]
                                pred_box = label_box[b][n]
                    if pred_cls > 0:
                        cls_out[b][i][j] = pred_cls
                        box_out[b][i][j] = pred_box

                else:
                    cls_out[b][i][j] = -1
    return torch.from_numpy(cls_out).long().to(device), torch.from_numpy(box_out).float().to(device)
