#! /usr/bin/env python

import os
import time
import argparse

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn as nn
from torch.autograd import Function
from torchvision import datasets, transforms
from torchvision.models import resnet

from torch2trt import torch2trt, TRTModule
#from testPR import torch2trt, TRTModule

def ResNet18(low_dim=128):
    net = resnet.ResNet(resnet.BasicBlock, [2, 2, 2, 2], low_dim)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                          stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    return net


def IDFD_trt(batch_size,fp16_mode,int8_mode = False):

    bs = batch_size

    model_path = "idfd_epoch_1999.pth"

    # モデルの定義
    low_dim = 128
    net = ResNet18(low_dim=low_dim).cuda().half().eval()

    print("check0")

    # 重みのロード
    net.load_state_dict(torch.load(model_path))

    print("check1")

    size = torch.randn((bs, 3, 32, 32)).cuda().half()

    print("check2")

    # model_trt = torch2trt(net, [size], fp16_mode=True, min_shapes=[(1, 3, 32, 32)], max_shapes=[(128, 3, 32, 32)], opt_shaps=[(64,3,32,32)])
    model_trt = torch2trt(net, [size], fp16_mode=fp16_mode,int8_mode = int8_mode, max_batch_size=bs)

    print("check3")

#     output_trt = model_trt(size)

#     output = net(size)

#     print(output.flatten()[0:10])
#     print(output_trt.flatten()[0:10])
#     print('max error: %f' % float(torch.max(torch.abs(output - output_trt))))

    torch.save(model_trt.state_dict(), f'IDFD_trt_{str(bs)}_int8.pth')

    model_trt = TRTModule()

    model_trt.load_state_dict(torch.load(f'IDFD_trt_{str(bs)}_int8.pth'))
    


if __name__=="__main__":
    
    fp16_mode = False
    int8_mode = True
    batch_list = [64,128,256,512]
    #batch_list = [1024,2048]
    
    for batch_size in batch_list:
        IDFD_trt(batch_size,fp16_mode,int8_mode)