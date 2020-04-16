# -*- coding: utf-8 -*-
# @Author : soda
#
# @License : Copyright © 2020 soda <soda791401863@gmail.com>
#
# @Contact : soda791401863@gmail.com
#
# @TIME : 2020-03-25 14:44:27
'''

'''
import torch
from torch.utils.data import DataLoader
import numpy as np
import json
import os

from model.fcos import Fcos
from API.Tester import Tester
from dataset.dataset import Data_Read
from util.eval_ap import eval_ap

with open('config.json', 'r') as f:
    cfg = json.load(f)

os.environ['CUDA_VISIBLE_DEVICES'] = cfg['CUDA_VISIBLE_DEVICES']

net = Fcos(cfg['pic_size'], cfg['class'], cfg['pic'], pretrained=False)
checkpoint = torch.load('net.pkl', map_location='cuda:%d' % cfg['device'][0])
net.load_state_dict(checkpoint['model'])
net = torch.nn.DataParallel(net, device_ids=cfg['device'])
net = net.cuda(cfg['device'][0])
net.eval()

test_data = Data_Read(cfg['test_path'], cfg['test_anno'], cfg['label_path'],
                      cfg['pic_size'], is_train=False)
test_loader = DataLoader(test_data, batch_size=cfg['test_batch'], shuffle=False,
                         num_workers=cfg['num_workers'], collate_fn=test_data.collate_fn)


tester = Tester(net, test_data, test_loader, cfg['nms_th'], cfg['nms_iou'])

tester.start()
