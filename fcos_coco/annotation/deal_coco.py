# -*- coding: utf-8 -*-
# @Author : soda
#
# @License : Copyright © 2020 soda <soda791401863@gmail.com>
#
# @Contact : soda791401863@gmail.com
#
# @TIME : 2020-03-26 22:03:39
'''

'''
from pycocotools.coco import COCO
import os
import json
import numpy as np
import pdb
from tqdm import tqdm

PATH = {'train': '/mnt/file/学习/代码/dataset/coco/annotations/instances_train2017.json',
        'val': '/mnt/file/学习/代码/dataset/coco/annotations/instances_val2017.json'}

for tv, path in PATH.items():
    coco = COCO(path)
    image_ids = coco.getImgIds()
    print(len(image_ids))
    cat_ids = coco.getCatIds()
    classes = coco.loadCats(cat_ids)

    coco_classes = {'background':0}  #  background:0
    coco_labels = {}  #  index:id
    coco_labels_inverse = {}  #  id:index
    label_name = []
    for cls in classes:
        coco_labels[len(coco_classes)] = cls['id']
        coco_labels_inverse[cls['id']] = len(coco_classes)
        coco_classes[cls['name']] = len(coco_classes)
    for name in coco_classes.keys():
        label_name.append(name)
    
    vaild_image_ids = []
    with open(tv+'.txt', 'w') as f:
        image_ids = tqdm(image_ids)
        for img_id in image_ids:
            line = []
            line.append(coco.loadImgs(img_id)[0]['file_name'])
            anno_ids = coco.getAnnIds(imgIds=img_id)
            annos = coco.loadAnns(anno_ids)
            boxes = []
            labels = []
            for idx, anno in enumerate(annos):
                if anno['bbox'][2] < 1 or anno['bbox'][3] < 1:
                    continue
                boxes.append(np.array(anno['bbox']))
                labels.append(np.array(coco_labels_inverse[anno['category_id']]))
            if len(boxes) <= 0:
                continue
            boxes = np.stack(boxes)
            labels = np.stack(labels)
            boxes[:, 2] += boxes[:, 0]
            boxes[:, 3] += boxes[:, 1]
            boxes = boxes.astype(np.int32)
            labels = labels.astype(np.int32).reshape((len(labels), 1))
            label_box = np.concatenate((labels, boxes), axis=1)
            for lb in label_box:
                line.append(' '.join([str(x) for x in lb]))
            f.write(','.join(line)+'\n')
            vaild_image_ids.append(img_id)
            image_ids.set_description()
    if tv is 'val':
        with open('label.txt', 'w') as f:
            for i in label_name:
                f.write(i+'\n')
        coco_dict = {
            'coco_labels': coco_labels,
            'coco_labels_inverse': coco_labels_inverse,
            'val_image_ids': vaild_image_ids
        }
        json.dump(coco_dict, open('coco_table.json', 'w'), indent=4)

