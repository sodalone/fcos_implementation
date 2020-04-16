# -*- coding: utf-8 -*-
# @Author : soda
#
# @License : Copyright © 2020 soda <soda791401863@gmail.com>
#
# @Contact : soda791401863@gmail.com
#
# @TIME : 2020-03-22 11:02:50
'''

'''

def get_lr(total_step, lr_param):
    warmup_step, warmup_factor, lr_base, lr_alpha, lr_schedule = lr_param
    lr = lr_base
    factor = warmup_factor[0] / warmup_factor[1]
    if total_step < warmup_step:
        x = float(total_step) / warmup_step
        a = 1 - factor
        b = factor
        lr = lr * (a*x+b) 
    else:
        for i in range(len(lr_schedule)):
            if total_step < lr_schedule[i]:
                break
            lr = lr * lr_alpha
    return lr

if __name__ == "__main__":
    print(get_lr(500, [500, [0.1, 3.0], 0.01, 0.1, [16000, 22000]]))