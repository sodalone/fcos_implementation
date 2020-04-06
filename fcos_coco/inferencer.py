# -*- coding: utf-8 -*-
# @Author : soda
#
# @License : Copyright © 2020 soda <soda791401863@gmail.com>
#
# @Contact : soda791401863@gmail.com
#
# @TIME : 2020-03-25 17:46:21
'''

'''

import torch
import json
import os
from PIL import Image

from model.fcos import Fcos
from API.Inferencer import Inferencer
from dataset.show import show_bbox

with open('config.json', 'r') as f:
    cfg = json.load(f)

os.environ['CUDA_VISIBLE_DEVICES'] = cfg['CUDA_VISIBLE_DEVICES']

net = Fcos(cfg['pic_size'], cfg['class'], cfg['pic'], pretrained=False)
checkpoint = torch.load('net.pkl', map_location='cuda:%d' % cfg['device'][0])
net.load_state_dict(checkpoint['model'])
net = net.cuda(cfg['device'][0])
net.eval()


nms_th = 0.4

label_name = []
with open(cfg['label_path'], 'r') as f:
    for line in f.readlines():
        label_name.append(line.strip())

inferencer = Inferencer(net)

for filename in os.listdir('images/'):
    if filename.endswith('jpg'):
        if filename[:5] == 'pred_':
            continue
        img = Image.open(os.path.join('images/', filename))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        pred_cls_i, pred_cls_s, pred_reg_i = inferencer.pred(img, cfg['pic_size'],
                                                              nms_th, cfg['nms_iou'])
        name = 'images/pred_'+filename.split('.')[0]+'.jpg'
        show_bbox(img, pred_reg_i.cpu(), pred_cls_i.cpu(), label_name, name, pred_cls_s.cpu())
