import torch
import numpy as np
import random
from IPython import display
from matplotlib import pyplot as plt
import torch.utils.data as Data
from PIL import Image
import os
from torch import nn
from mmcv.runner import BaseModule
import torch.optim as optim
from torch.nn import init
import torch.nn.functional as F
from ..builder import BACKBONES

@BACKBONES.register_module()
class Atrous(BaseModule):
    def __init__(self,outchannel=128,inchannel=128):
        super(Atrous,self).__init__()
        #定义六层卷积层
        #两层HDC（1,2,5,1,2,5）
        self.conv = nn.Sequential(
            #第一层 (3-1)*1+1=3 （64-3)/1 + 1 =62
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels = inchannel,out_channels = 64,kernel_size = 3 , stride = 1,padding='same',dilation=1),
            nn.BatchNorm2d(64),
            # inplace-选择是否进行覆盖运算
            nn.ReLU(inplace=False), 
            nn.Conv2d(in_channels = 64,out_channels = 32,kernel_size = 3 , stride = 1,padding='same',dilation=1),
            nn.BatchNorm2d(32),
            # inplace-选择是否进行覆盖运算
            nn.ReLU(inplace=False),
            #第二层 (3-1)*2+1=5 （62-5)/1 + 1 =58 
            nn.Conv2d(in_channels = 32,out_channels = 32,kernel_size = 3 , stride = 1,padding='same',dilation=2),
            nn.BatchNorm2d(32),
            # inplace-选择是否进行覆盖运算
            nn.ReLU(inplace=False),
            #第三层 (3-1)*5+1=11  (58-11)/1 +1=48
            nn.Conv2d(in_channels = 32,out_channels = 64,kernel_size = 3 , stride = 1,padding='same',dilation=5),
            nn.BatchNorm2d(64),
            # inplace-选择是否进行覆盖运算
            nn.ReLU(inplace=False),
             #第四层(3-1)*1+1=3 （48-3)/1 + 1 =46 
            nn.Conv2d(in_channels = 64,out_channels = 64,kernel_size = 3 , stride = 1,padding='same',dilation=1),
            nn.BatchNorm2d(64),
            # inplace-选择是否进行覆盖运算
            nn.ReLU(inplace=False),
            #第五层 (3-1)*2+1=5 （46-5)/1 + 1 =42 
            nn.Conv2d(in_channels = 64,out_channels = 64,kernel_size = 3 , stride = 1,padding='same',dilation=2),
            nn.BatchNorm2d(64),
            # inplace-选择是否进行覆盖运算
            nn.ReLU(inplace=False),
            #第六层 (3-1)*5+1=11  (42-11)/1 +1=32
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size = 3, stride = 1,padding='same',dilation=5),
            nn.BatchNorm2d(128),
            # inplace-选择是否进行覆盖运算
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=128, out_channels=outchannel, kernel_size = 3, stride = 1,padding='same'),
            nn.BatchNorm2d(outchannel),
            # inplace-选择是否进行覆盖运算
            nn.ReLU(inplace=False)
            
            
        )
        
    def forward(self,x):  
        ans = self.conv(x)
        # ans = F.avg_pool2d(ans,256)
        # ans = ans.squeeze()
        return ans 
