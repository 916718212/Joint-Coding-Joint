import warnings
import torch
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import BaseModule
from torch import nn as nn
from .aam import AxialAttention
from ..builder import BACKBONES
from .. import builder
from .rffm import Rffm
from .waspp import Atrous
import torch
import csv

def pairwise_euclidean_distance(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy - 2 * x @ y.t()
    # clamp()函数可以限定dist内元素的最大最小范围，dist最后开方，得到样本之间的距离矩阵
    dist = dist.clamp(min=1e-2).sqrt()
    dist = torch.nan_to_num(dist, nan=0, posinf=1e4)
    return dist


def calculate_entropy(feats, k=1):
    _,C,H,W = feats.size()
    feats = feats.view(-1,C,H*W)        
    N = C # N number dimension
    K = C//10 # 采用最合适的 k 值
    
    H_total = 0
    for feat in feats: # feat: (C,H*W)
        dist = pairwise_euclidean_distance(feat, feat) # (C, C)
        order = torch.argsort(dist, dim=1)
        for n in range(N):
            # ball V
            r_ball = dist[n][order[n][K]]
            H_total += r_ball
        # print(H_total)
    return torch.log(H_total+1)




@BACKBONES.register_module()
class SECOND(BaseModule):
    """Backbone network for SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (int): Input channels.
        out_channels (list[int]): Output channels for multi-scale feature maps.
        layer_nums (list[int]): Number of layers in each stage.
        layer_strides (list[int]): Strides of each stage.
        norm_cfg (dict): Config dict of normalization layers.
        conv_cfg (dict): Config dict of convolutional layers.
    """

    def __init__(self,
                 in_channels=128,
                 out_channels=[128, 128, 256],
                 layer_nums=[3, 5, 5],
                 layer_strides=[2, 2, 2],
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 conv_cfg=dict(type='Conv2d', bias=False),
                 init_cfg=None,
                 pretrained=None,
                 waspp=None,
                 attn1=None,
                 attn2=None
                 ):
        
        super(SECOND, self).__init__(init_cfg=init_cfg)
        assert len(layer_strides) == len(layer_nums)
        assert len(out_channels) == len(layer_nums)

        in_filters = [in_channels, *out_channels[:-1]]
        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        blocks = []
        
        self.waspp = waspp
        self.attn1 = attn1
        self.attn2 = attn2
        
        for i, layer_num in enumerate(layer_nums):
            block = [
                build_conv_layer(
                    conv_cfg,
                    in_filters[i],
                    out_channels[i],
                    3,
                    stride=layer_strides[i],
                    padding=1),
                build_norm_layer(norm_cfg, out_channels[i])[1],
                nn.ReLU(inplace=True),
            ]
            for j in range(layer_num):
                block.append(
                    build_conv_layer(
                        conv_cfg,
                        out_channels[i],
                        out_channels[i],
                        3,
                        padding=1))
                block.append(build_norm_layer(norm_cfg, out_channels[i])[1])
                block.append(nn.ReLU(inplace=True))

            block = nn.Sequential(*block)
            blocks.append(block)

        self.conv=nn.Conv2d(in_channels = 256,out_channels = 256,kernel_size = 3)
        
        self.blocks = nn.ModuleList(blocks)

        
        if waspp:
            self.waspp = builder.build_backbone(waspp)
        if attn1:
            self.attn1 = builder.build_backbone(attn1)
        if attn2:
            self.attn2 = builder.build_backbone(attn2)  

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        else:
            self.init_cfg = dict(type='Kaiming', layer='Conv2d')

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input with shape (N, C, H, W).

        Returns:
            tuple[torch.Tensor]: Multi-scale features.
        """
        outs = []
        h1=calculate_entropy(x)
        z1 = self.blocks[0](x)
        x = self.blocks[0](x) 
        y1 = self.attn1(x)
        #print('attn1-output:',y1.shape)
        h2=calculate_entropy(z1)
        outs.append(y1)
        t = self.waspp(z1)
        z2 = self.blocks[1](t)         
        y2 = self.attn2(z2)
        h3=calculate_entropy(z2)
        #print('attn2-output:',y2.shape)
        outs.append(y2)
        var=0
        delta_entropy=[]
        delta_h1=h1-h2
        delta_h2=h2-h3
        delta_entropy.append(delta_h1)
        delta_entropy.append(delta_h2)
        delta_entropy = torch.stack(delta_entropy)
        var += torch.var(delta_entropy) 
        eloss=[]
        var_numpy=var.cpu().detach().numpy()
        eloss.append(var_numpy)
        path='/root/autodl-tmp/joint-eloss-second.csv'

        with open(path,'a+') as f:
            csv_write = csv.writer(f)
            csv_write.writerow(eloss)
       

        return tuple(outs)