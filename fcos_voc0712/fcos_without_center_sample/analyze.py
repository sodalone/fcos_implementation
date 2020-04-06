# -*- coding: utf-8 -*-
# @Author : soda
#
# @License : Copyright © 2020 soda <soda791401863@gmail.com>
#
# @Contact : soda791401863@gmail.com
#
# @TIME : 2020-03-25 14:44:38
'''

'''
import numpy as np
import matplotlib.pyplot as plt

log = np.load('log.npy', allow_pickle=True)

map_mean, map_50, map_75, loss = [], [], [], []

for i in range(len(log)):
    map_mean.append(log[i][0])
    map_50.append(log[i][1])
    map_75.append(log[i][2])
    loss.extend(log[i][3])

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.plot(map_mean, color='r', label='map_mean')
ax1.plot(map_50, color='b', label='map_50')
ax1.plot(map_75, color='g', label='map_75')
ax1.legend()
ax2.plot(loss, color='black', label='loss')
ax2.legend()

map_mean = np.around(map_mean, decimals=4)
map_50 = np.around(map_50, decimals=4)
map_75 = np.around(map_75, decimals=4)

print('No', '\t', 'map_mean', '\t', 'map_50', '\t', 'map_75')
for i in range(len(map_mean)):
    print('%-9d%-9.4f\t%-9.4f\t%-9.4f' % (i, map_mean[i], map_50[i], map_75[i]))

plt.show()