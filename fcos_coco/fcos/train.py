# -*- coding: utf-8 -*-
# @Author : soda
#
# @License : Copyright © 2020 soda <soda791401863@gmail.com>
#
# @Contact : soda791401863@gmail.com
#
# @TIME : 2020-03-22 10:33:53
'''

'''

import torch
from torch.utils.data import DataLoader
import numpy as np

import json
import os
import random

from dataset.dataset import Data_Read
from model.fcos import Fcos
from API.Trainer import Trainer
from API.Tester import Tester

with open('config.json', 'r') as f:
    cfg = json.load(f)

os.environ['CUDA_VISIBLE_DEVICES'] = cfg['CUDA_VISIBLE_DEVICES']

if cfg['seed'] >= 0:
    seed = cfg['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU

net = Fcos(cfg['pic_size'], cfg['class'], cfg['pic'], cfg['pretrained'])
optimizer = torch.optim.SGD(net.parameters(), lr=cfg['lr_base'],
                            momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
                       
log = []
checkpoint = None

if cfg['load']:
    checkpoint = torch.load('net.pkl', map_location='cuda:%d' % cfg['device'][0])
    log = list(np.load('log.npy', allow_pickle=True))
    net.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("读取成功")

net = torch.nn.DataParallel(net, device_ids=cfg['device'])
net = net.cuda(cfg['device'][0])
net.train()

for state in optimizer.state.values():
    for i, name in state.items():
        if isinstance(name, torch.Tensor):
            state[i] = name.cuda()

train_data = Data_Read(cfg['train_path'], cfg['train_anno'], cfg['label_path'],
                       cfg['pic_size'], True, cfg['is_normalize'], cfg['arae_th'],
                       cfg['img_minscale'], cfg['is_augmentation'])

test_data = Data_Read(cfg['test_path'], cfg['test_anno'], cfg['label_path'],
                      cfg['pic_size'], is_train=False)

train_loader = DataLoader(train_data, batch_size=cfg['train_batch'], shuffle=True,
                          num_workers=cfg['num_workers'], collate_fn=train_data.collate_fn)

test_loader = DataLoader(test_data, batch_size=cfg['test_batch'], shuffle=False,
                         num_workers=cfg['num_workers'], collate_fn=test_data.collate_fn)



lr_param = [cfg['warmup_step'], cfg['warmup_factor'],
            cfg['lr_base'], cfg['lr_alpha'], cfg['lr_schedule']]

trainer = Trainer(net, train_data, train_loader, optimizer, cfg['grad_clip'], lr_param)
tester = Tester(net, test_data, test_loader, cfg['nms_th'], cfg['nms_iou'])

if cfg['load']:
    trainer.step = checkpoint['step']
    trainer.epoch = checkpoint['epoch']


while trainer.epoch < cfg['epoch']:

    net.module.backbone.freeze_stages(cfg['freeze_stage'])
    if cfg['freeze_bn']:
        net.module.backbone.freeze_bn()

    loss_list = trainer.start()
    map_mean, map_50, map_75 = tester.start()
    log.append([map_mean, map_50, map_75, loss_list])
    if cfg['save']:
        torch.save({
            'model': net.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': trainer.epoch,
            'step': trainer.step
        }, 'net.pkl')
        np.save('log.npy', log)
        print("保存成功 epoch: %d" % (trainer.epoch-1))
print('finished!')
