# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 22:18:12 2019

@author: lenovo
"""

import torch
import torch.nn as nn





#
def double_conv_0(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=False),
#        nn.LeakyReLU(inplace=True),

        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=False),
#        nn.LeakyReLU(inplace=True),

        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=False),
#        nn.LeakyReLU(inplace=True),

    )

def double_conv_1(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=False),
#        nn.LeakyReLU(inplace=True),

        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=False),
#        nn.LeakyReLU(inplace=True),

    )

def double_conv_2(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=False),
#        nn.LeakyReLU(inplace=True),

        nn.Conv2d(out_channels, in_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=False),
#        nn.LeakyReLU(inplace=True),

    )

def double_conv_3(in_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels//2, 3, padding=1),
        nn.BatchNorm2d(in_channels//2),
        nn.ReLU(inplace=False),
#        nn.LeakyReLU(inplace=True),

        nn.Conv2d(in_channels//2, in_channels//4, 3, padding=1),
        nn.BatchNorm2d(in_channels//4),
        nn.ReLU(inplace=False),
#        nn.LeakyReLU(inplace=True),

    )

def decon(in_channels):
    return nn.Sequential(
       nn.ConvTranspose2d(in_channels, in_channels//2, 3, 2, 1, 1),
       nn.BatchNorm2d(in_channels//2),
       nn.ReLU(inplace=False),

            )

class UNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.dconv_down1 = double_conv_0(1, 64//2)
        self.dconv_down2 = double_conv_1(64//2, 128//2)
        self.dconv_down3 = double_conv_1(128//2, 256//2)
        self.dconv_down4 = double_conv_1(256//2, 512//2)
        self.dconv_down5_new = double_conv_1(512//2, 1024//2)
#        self.dconv_down5 = double_conv_2(512, 1024)   #SY

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_new1 = decon(1024//2)
        self.upsample_new2 = decon(512//2)
        self.upsample_new3 = decon(256//2)
        self.upsample_new4 = decon(128//2)

#        self.dconv_up4 = double_conv_3(512 + 512)
#        self.dconv_up3 = double_conv_3(256 + 256)
#        self.dconv_up2 = double_conv_3(128 + 128)
#        self.dconv_up1 = double_conv_1(64 + 64, 64)
        self.dconv_up4 = double_conv_1(1024//2, 512//2)
        self.dconv_up3 = double_conv_1(512//2, 256//2)
        self.dconv_up2 = double_conv_1(256//2, 128//2)
        self.dconv_up1 = double_conv_1(128//2, 64//2)

        self.conv_last = nn.Conv2d(64//2, 1, 1)


    def forward(self, x0):

        conv1 = self.dconv_down1(x0)         #128*128*64
        x = self.maxpool(conv1)             #64*64*64

        conv2 = self.dconv_down2(x)         #64*64*128
        x = self.maxpool(conv2)             #32*32*128

        conv3 = self.dconv_down3(x)         #32*32*256
        x = self.maxpool(conv3)             #16*16*256

        conv4 = self.dconv_down4(x)         #16*16*512
        x = self.maxpool(conv4)         #SY #8*8*512

#        x = self.dconv_down5(x)         #SY #8*8*512
        x = self.dconv_down5_new(x)         #SY #8*8*1024

#        x = self.upsample(x)                #16*16*512
        x = self.upsample_new1(x)           #16*16*512
        x = torch.cat([conv4, x], dim=1)    #16*16*1024

        x = self.dconv_up4(x)               #16*16*512
        x = self.upsample_new2(x)           #32*32*256
        x = torch.cat([conv3, x], dim=1)    #32*32*512

        x = self.dconv_up3(x)               #32*32*256
        x = self.upsample_new3(x)           #64*64*128
        x = torch.cat([conv2, x], dim=1)    #64*64*256

        x = self.dconv_up2(x)               #64*64*128
        x = self.upsample_new4(x)           #128*128*64
        x = torch.cat([conv1, x], dim=1)    #128*128*128

        x = self.dconv_up1(x)               #128*128*64

        x = self.conv_last(x)             #128*128*1
        out = x+x0

        return out
