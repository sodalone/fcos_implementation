# -*- coding: utf-8 -*-
# @Author : soda
#
# @License : Copyright © 2020 soda <soda791401863@gmail.com>
#
# @Contact : soda791401863@gmail.com
#
# @TIME : 2020-03-16 14:06
'''
backbone网络使用ResNet50
'''

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import model_zoo
import math

model_url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, outplanes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, outplanes,
                               kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.conv2 = nn.Conv2d(
            outplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.conv3 = nn.Conv2d(
            outplanes, outplanes*self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes*self.expansion)

        self.downsample = nn.Sequential()
        if stride != 1 or inplanes != outplanes*self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, outplanes*self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outplanes*self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out, inplace=True)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes * block.expansion
        for i in range(blocks-1):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out3 = self.layer2(out)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)

        # out3 scale 8    out4 scale 16    out sclae 32
        return (out3, out4, out5)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def freeze_stages(self, stage):
        if stage >= 0:
            for m in [self.conv1, self.bn1]:
                for param in m.parameters():
                    param.requires_grad = False
        for i in range(1, stage+1):
            m = getattr(self, 'layer{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False


def resnet50(pretrained=False):
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        try:
            model.load_state_dict(torch.load(
                '../../pretrained_model/resnet50-19c8e357.pth'), strict=False)
        except:
            model.load_state_dict(model_zoo.load_url(
                model_url, '../../pretrained_model/'), strict=False)
    return model


if __name__ == "__main__":
    a, b, c = resnet50()(torch.randn(1, 3, 641, 641))
    print(a.shape, b.shape, c.shape)
