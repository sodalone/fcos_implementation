# -*- coding: utf-8 -*-
# @Author : soda
#
# @License : Copyright © 2020 soda <soda791401863@gmail.com>
#
# @Contact : soda791401863@gmail.com
#
# @TIME : 2020-03-26 11:15:17
'''

'''

def BCE_loss(pred, target):
    pred = pred.sigmoid()
    loss = -1 * (pred.log()*target + (1.0-pred).log()*(1.0-target))
    return loss