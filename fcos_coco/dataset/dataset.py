# -*- coding: utf-8 -*-
# @Author : soda
#
# @License : Copyright © 2020 soda <soda791401863@gmail.com>
#
# @Contact : soda791401863@gmail.com
#
# @TIME : 2020-03-19 21:35:02
'''
数据读入处理
'''
import torch
from torch.utils import data
import torchvision
import numpy as np
from PIL import Image


import os
import math
import random


class Data_Read(data.Dataset):

    def __init__(self, data_path, data_anno, label_path, size,
                 is_train, is_normalize=True, area_th=35, img_minscale=0.6, is_augmentation=True):
        self.data_path = data_path
        self.names = []
        self.boxes = []
        self.labels = []
        self.label_name = []
        self.is_train = is_train
        self.is_normalize = is_normalize
        self.is_augmentation = is_augmentation
        self.area_th = area_th
        self.size = size
        self.img_minscale = img_minscale

        self.normalizer = torchvision.transforms.Normalize((0.485, 0.456, 0.406),
                                                           (0.229, 0.224, 0.225))

        with open(data_anno, 'r') as f:
            lines = f.readlines()
            self.num = lines.__len__()

            for line in lines:
                name, label_box = line.strip().split(',', 1)
                self.names.append(name)
                label_box_ = label_box.split(',')
                box = []
                label = []
                for i in label_box_:
                    splits = i.split()
                    box.append([j for j in map(float, splits[1:])])
                    label.append(int(splits[0]))
                self.boxes.append(torch.FloatTensor(box))
                self.labels.append(torch.LongTensor(label))

        with open(label_path, 'r') as f:
            for line in f.readlines():
                self.label_name.append(line.strip())

    def __len__(self):
        return self.num

    def __getitem__(self, index):

        img = Image.open(os.path.join(self.data_path, self.names[index]))
        if img.mode != "RGB":
            img = img.convert("RGB")

        img_w, img_h = img.size
        box = self.boxes[index].clone()
        label = self.labels[index].clone()
        # 保证box范围
        box[:, :2].clamp_(min=1)
        box[:, 2].clamp_(max=img_w-1)
        box[:, 3].clamp_(max=img_h-1)

        if self.is_train:
            if self.is_augmentation:
                img, box = flip(img, box, random.random())
                img, box = colorJitter(img, box, random.random())
            if random.random() < 0.5:
                img, box, local, scale = random_fix(img, box, self.size, self.img_minscale)
            else:
                img, box, local, scale = center_fix(img, box, self.size)

        else:
            img, box, local, scale = center_fix(img, box, self.size)

        h_w = box[:, 2:] - box[:, :2]
        area = h_w[:, 0] * h_w[:, 1]  # 求出面积
        flag_area = area >= self.area_th
        box, label = box[flag_area], label[flag_area]
        img = torchvision.transforms.ToTensor()(img)

        if self.is_normalize:
            img = self.normalizer(img)

        return img, box, label, local, scale

    def collate_fn(self, data):
        # datalodaer 读取
        img, box, label, local, scale = zip(*data)
        img_t = torch.stack(img, dim=0)
        batch_size = len(box)
        n_max = 0
        for i in box:
            n = i.shape[0]
            n_max = n if n > n_max else n_max

        box_t = torch.zeros(batch_size, n_max, 4)
        label_t = torch.zeros(batch_size, n_max).long()
        for i in range(batch_size):
            box_t[i, 0:box[i].shape[0]] = box[i]
            label_t[i, 0:box[i].shape[0]] = label[i]
        local_t = torch.stack(local, dim=0)
        scale_t = torch.FloatTensor(scale)

        return img_t, box_t, label_t, local_t, scale_t


# 数据增强，翻转
def flip(img, box, random):
    if random < 0.25:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        w = img.size[0]
        if box.shape[0] != 0:
            xmin = w - box[:, 2]
            xmax = w - box[:, 0]
            box[:, 0] = xmin
            box[:, 2] = xmax
        return img, box
    elif random >= 0.25 and random < 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        h = img.size[1]
        if box.shape[0] != 0:
            ymin = h - box[:, 3]
            ymax = h - box[:, 1]
            box[:, 1] = ymin
            box[:, 3] = ymax
        return img, box
    else:
        return img, box


# 数据增强 亮度对比度饱和度变换
def colorJitter(img, box, random, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1):
    if random < 0.3:
        img = torchvision.transforms.ColorJitter(brightness=brightness,
                                             contrast=contrast, saturation=saturation, hue=hue)(img)
    return img, box


def center_fix(img, box, size):
    w, h = img.size
    size_max = max(w, h)
    scale = float(size) / size_max
    w_fix, h_fix = int(w * scale + 0.5), int(h * scale + 0.5)
    w_ofst, h_ofst = round((size-w_fix) / 2.0), round((size-h_fix) / 2.0)

    img = img.resize((w_fix, h_fix), Image.BILINEAR)
    img = img.crop((-w_ofst, -h_ofst, size-w_ofst, size-h_ofst))

    if box.shape[0] != 0:
        box = box * torch.FloatTensor([scale]*4)
        box += torch.FloatTensor([w_ofst, h_ofst]*2)
    local = torch.FloatTensor([w_ofst, h_ofst, w_fix+w_ofst, h_fix+h_ofst])

    return img, box, local, scale


def random_fix(img, box, size, img_minscale):
    w, h = img.size
    size_max = max(w, h)
    scale = float(size) / size_max * random.uniform(img_minscale, 1.0)
    w_fix, h_fix = int(w * scale + 0.5), int(h * scale + 0.5)
    w_ofst = random.randint(0, size-w_fix)
    h_ofst = random.randint(0, size-h_fix)
    img = img.resize((w_fix, h_fix), Image.BILINEAR)
    img = img.crop((-w_ofst, -h_ofst, size-w_ofst, size-h_ofst))

    if box.shape[0] != 0:
        box = box * torch.FloatTensor([scale]*4)
        box += torch.FloatTensor([w_ofst, h_ofst]*2)
    local = torch.FloatTensor([w_ofst, h_ofst, w_fix+w_ofst, h_fix+h_ofst])

    return img, box, local, scale


if __name__ == "__main__":
    from show import show_bbox
    
    train = True
    size = 641
    area_th = 32
    img_minscale = 0.6
    is_augmentation = False
    batch_size = 8
    csv_root = '/mnt/file/学习/代码/dataset/pascal_voc_2012/VOC0712/VOC_trainval/JPEGImages'
    csv_list = 'annotation/trainval.txt'
    csv_name = 'annotation/label.txt'

    dataset = Data_Read(csv_root, csv_list, csv_name,
                        size=size, is_train=train, is_normalize=False, area_th=area_th,
                        img_minscale=img_minscale, is_augmentation=is_augmentation)
    dataloader = data.DataLoader(dataset, batch_size=batch_size,
                                 shuffle=True, num_workers=0, collate_fn=dataset.collate_fn)
    for imgs, boxes, labels, locs, scales in dataloader:
        print(imgs.shape)
        print(boxes.shape)
        print(labels.shape)
        print(locs.shape)
        for i in range(len(boxes)):
            print(i, ': ', boxes[i].shape, labels[i].shape, locs[i])
        idx = 0
        print(labels[idx])
        print(boxes[idx][labels[idx] > 0])
        show_bbox(imgs[idx], boxes[idx], labels[idx], dataset.label_name)
