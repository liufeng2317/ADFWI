'''
* Author: LiuFeng(USTC) : liufeng2317@mail.ustc.edu.cn
* Date: 2024-05-05 19:51:52
* LastEditors: LiuFeng
* LastEditTime: 2024-05-05 20:29:15
* Description: 
* Copyright (c) 2024 by liufeng, Email: liufeng2317@mail.ustc.edu.cn, All Rights Reserved.
'''
# https://pytorch.org/docs/stable/optim.html
from torch.optim import Adadelta,Adagrad,Adam,AdamW,Adamax,ASGD,NAdam,RAdam,Rprop,SGD

from ncg_optimizer import LCG, BASIC as NLCG

from .optimizer_swit import Swit_Optimizer
