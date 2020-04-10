# -*- coding: utf-8 -*-
# @Author : soda
#
# @License : Copyright © 2020 soda <soda791401863@gmail.com>
#
# @Contact : soda791401863@gmail.com
#
# @TIME : 2020-03-20 10:41:05
'''
画框
'''

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from torchvision import transforms

COLOR_TABLE = ['Red'] * 100

def draw_bbox_text(drawObj, xmin, ymin, xmax, ymax, text, color, bd=2):
    drawObj.rectangle((xmin, ymin, xmax, ymin+bd), fill=color)
    drawObj.rectangle((xmin, ymax-bd, xmax, ymax), fill=color)
    drawObj.rectangle((xmin, ymin, xmin+bd, ymax), fill=color)
    drawObj.rectangle((xmax-bd, ymin, xmax, ymax), fill=color)
    strlen = len(text)
    drawObj.rectangle((xmin, ymin, xmin+strlen*6+5, ymin+12), fill=color)
    drawObj.text((xmin+3, ymin), text)


def show_bbox(img, boxes, labels, NAME_TAB, file_name=None, scores=None,
              matplotlib=True, lb_g=True):
    if lb_g:
        bg_idx = 0
    else:
        bg_idx = -1
    if not isinstance(img, Image.Image):
        img = transforms.ToPILImage()(img)
    drawObj = ImageDraw.Draw(img)

    for box_id in range(boxes.shape[0]):
        label = int(labels[box_id])
        if label > bg_idx:
            box = boxes[box_id]
            if scores is None:
                draw_bbox_text(drawObj, box[0], box[1], box[2], box[3], NAME_TAB[label],
                               color=COLOR_TABLE[label])
            else:
                str_score = str(float(scores[box_id]))[:5]
                str_out = NAME_TAB[label] + ': ' + str_score
                draw_bbox_text(drawObj, box[0], box[1], box[2], box[3], str_out,
                               color=COLOR_TABLE[label])
    if file_name is not None:
        img.save(file_name)
    else:
        if matplotlib:
            plt.imshow(img, aspect='equal')
            plt.show()
        else:
            img.show()
