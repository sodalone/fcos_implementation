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

ap_mean, ap_50, ap_75, loss = [], [], [], []

for i in range(len(log)):
    ap_mean.append(log[i][0])
    ap_50.append(log[i][1])
    ap_75.append(log[i][2])
    loss.extend(log[i][3])

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.plot(ap_mean, color='r', label='ap_mean')
ax1.plot(ap_50, color='b', label='ap_50')
ax1.plot(ap_75, color='g', label='ap_75')
ax1.legend()
ax2.plot(loss, color='black', label='loss')
ax2.legend()

ap_mean = np.around(ap_mean, decimals=4)
ap_50 = np.around(ap_50, decimals=4)
ap_75 = np.around(ap_75, decimals=4)

print('No', '\t', 'ap_mean', '\t', 'ap_50', '\t\t', 'ap_75')
for i in range(len(ap_mean)):
    print('%-9d%-9.4f\t%-9.4f\t%-9.4f' % (i, ap_mean[i], ap_50[i], ap_75[i]))

plt.show()