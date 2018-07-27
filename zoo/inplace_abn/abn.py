import torch
import torch.nn as nn
import sys

#from modules import InPlaceABN


class ABN(nn.Module):
    def __init__(self, in_chs, activation_fn=nn.ReLU(inplace=True)):
        super(ABN, self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_chs, eps=0.001),
            activation_fn
        )

    def forward(self, x):
        return self.layer(x)


#class InABN(nn.Module):
#    def __init__(self, in_chs, activation_fn=nn.ReLU(inplace=True)):
#        super(InABN, self).__init__()
#        self.layer = InPlaceABN(in_chs, eps=0.001, activation='relu')
#
#    def forward(self, x):
#        return self.layer(x)
