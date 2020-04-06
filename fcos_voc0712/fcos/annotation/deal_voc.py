# -*- coding: utf-8 -*-
# @Author : soda
#
# @License : Copyright © 2020 soda <soda791401863@gmail.com>
#
# @Contact : soda791401863@gmail.com
#
# @TIME : 2020-03-14 21:29

'''
该文件用于处理VOC0712原始数据中的annotation，以让程序读入
'''

import os
import xml.etree.ElementTree as ET
from tqdm import tqdm

TRAINVAL_PATH = [
    'trainval', '/mnt/file/学习/代码/dataset/pascal_voc_2012/VOC0712/VOC_trainval']
TEST_PATH = ['test', '/mnt/file/学习/代码/dataset/pascal_voc_2012/VOC0712/VOC_test']
CODE_PATH = '/mnt/file/学习/毕业设计/fcos_implementation/fcos'

VOC_LABEL = {
    'background': '0',
    'aeroplane': '1',
    'bicycle': '2',
    'bird': '3',
    'boat': '4',
    'bottle': '5',
    'bus': '6',
    'car': '7',
    'cat': '8',
    'chair': '9',
    'cow': '10',
    'diningtable': '11',
    'dog': '12',
    'horse': '13',
    'motorbike': '14',
    'person': '15',
    'pottedplant': '16',
    'sheep': '17',
    'sofa': '18',
    'train': '19',
    'tvmonitor': '20'
}

for voc_type, path in [TRAINVAL_PATH, TEST_PATH]:
    annotations_path = os.path.join(path, 'Annotations')
    annotations = os.listdir(annotations_path)
    with open(os.path.join(CODE_PATH, 'annotation', voc_type+'.txt'), 'w') as f:
        annotations = tqdm(annotations)
        for anno in annotations:
            line = []
            name, _ = anno.split('.')
            line.append(name+'.jpg')

            anno_path = os.path.join(annotations_path, anno)
            tree = ET.parse(anno_path)
            root = tree.getroot()
            objects = root.findall('object')

            for ob in objects:
                place = []

                if voc_type == 'test' and int(ob.find('difficult').text) == 1:
                    continue
                label = ob.find('name').text
                index = str(VOC_LABEL[label])
                place.append(index)
                bndbox = ob.find('bndbox')

                for tag in ['xmin', 'ymin', 'xmax', 'ymax']:
                    num = str(eval(bndbox.find(tag).text)-1)
                    place.append(num)
                line.append(' '.join(place))
            f.write(','.join(line)+'\n')
            annotations.set_description()

with open(os.path.join(CODE_PATH, 'annotation', 'label.txt'), 'w') as f:
    labels = VOC_LABEL.keys()
    f.write('\n'.join(labels))
    print('label.txt 制作完成，共计{}种'.format(labels.__len__()))
