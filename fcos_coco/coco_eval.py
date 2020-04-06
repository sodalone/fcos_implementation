# -*- coding: utf-8 -*-
# @Author : soda
#
# @License : Copyright © 2020 soda <soda791401863@gmail.com>
#
# @Contact : soda791401863@gmail.com
#
# @TIME : 2020-03-29 16:39:28
'''

'''
import torch
import json
import os
from pycocotools.coco import  COCO
from pycocotools.cocoeval import COCOeval

from dataset.dataset import Data_Read
from model.fcos import Fcos
from API.COCO_eval import COCO_eval

with open('config.json', 'r') as f:
    cfg = f.load(f)

with open(cfg['coco_table'], 'r') as f:
    coco_table = f.load(f)

os.environ['CUDA_VISIBLE_DEVICES'] = cfg['CUDA_VISIBLE_DEVICES']

net = Fcos(cfg['pic_size'], cfg['class'], cfg['pic'], pretrained=False)
checkpoint = torch.load('net.pkl', map_location='cuda:%d' % cfg['device'][0])
net.load_state_dict(checkpoint['model'])
net = net.cuda(cfg['device'][0])
net.eval()

test_data = Data_Read(cfg['test_path'], cfg['test_anno'], cfg['label_path'],
                      cfg['pic_size'], is_train=False)

coco_evaler = COCO_eval(net, test_data, coco_table['val_image_ids'], coco_table['coco_labels'])

coco_evaler.start()
coco = COCO(cfg['coco_anno'])
coco_pred = coco.loadRes('coco_bbox_results.json')
coco_eval = COCOeval(coco, coco_pred, 'bbox')
coco_eval.params.imgIds = coco_table['val_image_ids']
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

