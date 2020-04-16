# -*- coding: utf-8 -*-
# @Author : soda
#
# @License : Copyright © 2020 soda <soda791401863@gmail.com>
#
# @Contact : soda791401863@gmail.com
#
# @TIME : 2020-03-23 20:51:21
'''

'''

import torch
from time import time
import numpy as np

from util.get_lr import get_lr
from model.fcos import get_pred


class Trainer(object):
    def __init__(self, net, train_data, train_loader, optimizer, grad_clip, lr_param):
        self.net = net
        self.train_data = train_data
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.grad_clip = grad_clip
        self.lr_param = lr_param
        self.epoch = 0
        self.step = 0

    def start(self):
        all_batch_size = 0
        loss_list = []
        for imgs, boxes, labels, locals, scales in self.train_loader:
            lr = get_lr(self.step, self.lr_param)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            batch_size = imgs.shape[0]
            all_batch_size += batch_size
            start_time = time()
            self.optimizer.zero_grad()
            loss = self.net(imgs, locals, labels, boxes)
            loss = torch.mean(loss)
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.net.parameters(), self.grad_clip)
            self.optimizer.step()
            cost_time = (time()-start_time)*1000
            self.step += 1
            print('total_step:%d\tepoch:%d\t\tstep:%d/%d\t\tloss:%f\tlr:%f\ttime:%dms' %
                  (self.step, self.epoch, all_batch_size, len(self.train_data),
                   loss, lr, cost_time))
            loss_list.append(loss)
        self.epoch += 1
        return loss_list
