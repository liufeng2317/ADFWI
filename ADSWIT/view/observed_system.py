'''
* Author: LiuFeng(USTC) : liufeng2317@mail.ustc.edu.cn
* Date: 2024-04-19 16:55:27
* LastEditors: LiuFeng
* LastEditTime: 2024-05-13 10:57:46
* Description: 
* Copyright (c) 2024 by liufeng, Email: liufeng2317@mail.ustc.edu.cn, All Rights Reserved.
'''
import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits import axes_grid1

import warnings
warnings.filterwarnings("ignore")

def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

# def plot_observeSystem(param,model,src,rcv,save_path="",show=False):
#     v = model.v
#     rcv_x = rcv.rcv_x
#     rcv_z = rcv.rcv_z
#     src_x = src.src_x
#     src_z = src.src_z
#     pml = param.pml
    
#     fig = plt.figure(figsize=(12,8))
#     ax = plt.axes()
#     # im = plt.imshow(v,cmap='jet_r',vmin=3000,vmax=3030)
#     im = plt.imshow(v,cmap='jet_r')
#     cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
#     # plt.colorbar(im,cax=cax)
#     plt.colorbar(im,cax=cax)
#     ax.scatter(rcv_x,rcv_z,20,marker="v",c='w',label="receiver")
#     ax.scatter(src_x-pml,src_z-pml,20,marker='*',c='k',label="source")
#     ax.legend(fontsize=12)
#     if save_path != "":
#         plt.savefig(save_path,bbox_inches="tight")
#     if show:
#         plt.show()
#     else:
#         plt.close()