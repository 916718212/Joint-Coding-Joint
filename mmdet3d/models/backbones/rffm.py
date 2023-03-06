import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from ..builder import BACKBONES

@BACKBONES.register_module()
class Rffm(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels):
        super(Rffm, self).__init__()
        self.conv1 = nn.Conv2d(in_channels1, int(out_channels/2), kernel_size=(3,3))
        self.conv2 = nn.Conv2d(in_channels2, int(out_channels/2), kernel_size=(3, 3))
        self.conv3 = nn.Conv2d(int(out_channels/2), int(out_channels/2), kernel_size=(3,3))
        self.pooling = nn.MaxPool2d(kernel_size=(5,5))
        #self.fc = nn.Linear(320, 10)
        self.relu = nn.ReLU()
        self.bn = torch.nn.BatchNorm2d(int(out_channels/2))

    def residential_unit(self, s):
        s1 = self.relu(s)
        s2 = self.relu(s)
        s2 = self.pooling(s2)
        print('b', s2.shape)
        s2 = self.conv3(s2)
        #print(s2.shape)
        h1, w1 = tuple(s1.shape)[2], tuple(s1.shape)[3]
        h2, w2 = tuple(s2.shape)[2], tuple(s2.shape)[3]
        #s2 = s2.expand(tuple(s1.shape))
        s2 = F.pad(s2, [0, h1-h2, 0, w1-w2])
        print(s1.shape)
        print(s2.shape)
        s3 = s1 + s2
        #s3 = torch.cat((s1, s2), dim=1)
        return s3

    def normal_unit1(self, s):
        s = self.conv1(s)
        s = self.bn(s)
        s = self.relu(s)
        #print('a', s.shape)
        return s

    def normal_unit2(self, s):
        s = self.conv2(s)
        s = self.bn(s)
        s = self.relu(s)
        #print('a', s.shape)
        return s

    def forward(self, x1, x2):
        x1 = self.normal_unit1(x1)
        x2 = self.normal_unit2(x2)
        x3 = torch.matmul(x1, x2)
        x3 = self.residential_unit(x3)

        return x3


# r = rffm(num_channels=8, out_channels=8)
# l1 = np.random.randint(1, 10, size=(1, 8, 30, 30))
# l2 = np.random.randint(11, 20, size=(1, 4, 30, 30))
# #print(l)
# #l1 = torch.from_numpy(l1)
# #print(l1)
# l1 = torch.FloatTensor(l1)
# #print(l1)
# l2 = torch.FloatTensor(l2)
# #print(l2)


# #l4 = r.residential_unit(l1)
# #print(l4)

# l5 = r.forward(l1, l2)
# print(l5)
