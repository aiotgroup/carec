import torch
import torch.nn as nn
import torch.nn.functional as F


import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool1d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, linear=False):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if linear:
            self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        else:
            self.up = nn.ConvTranspose1d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # x1 = F.interpolate(x1, scale_factor=2, mode='linear', align_corners=True)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffY // 2, diffY - diffY//2))
        
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_channels=52, n_classes=7):
        super(UNet, self).__init__()
        self.conv = nn.Conv1d(n_channels, n_channels, kernel_size=3,stride=2, bias=False)
#         self.relu1 = nn.ReLU()
#         self.bn1 = nn.BatchNorm1d(out_channel)
        self.inc = inconv(n_channels, 128)
        self.down1 = down(128, 256)
        self.down2 = down(256, 512)
        self.down3 = down(512, 1024)
        self.down4 = down(1024, 1024)
        self.up1 = up(2048, 512)
        self.up2 = up(1024, 256)
        self.up3 = up(512, 128)
        self.up4 = up(256, 128)
        self.outc = outconv(128, 512)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = 128

    def forward(self, x):
        # x = self.conv(x)
        # print(x.shape)        # [bs, 270, 1000]
        x1 = self.inc(x)        # [bs, 128, 1000]
        # print(x1.shape)
        x2 = self.down1(x1)     # [bs, 256，500]
        # print(x2.shape)
        x3 = self.down2(x2)     # [bs, 512，250]
        # print(x3.shape)
        x4 = self.down3(x3)     # [bs, 1024，125]
        # print(x4.shape)
        x5 = self.down4(x4)     # [bs, 1024，62]
        # print(x5.shape)
        x = self.up1(x5, x4)    # [bs, 512，125]
        # print(x.shape)
        x = self.up2(x, x3)     # [bs, 256，250]
        # print(x.shape)
        x = self.up3(x, x2)     # [bs, 128, 500]
        # print(x.shape)
        x = self.up4(x, x1)     # [bs, 128, 1000]
        # print(x.shape)
        # x = self.outc(x)    
        # print(x.shape)
        x = self.avgpool(x)     # [bs,128,1]
        # print(x.shape)
        x = x.mean(dim=-1)      # [bs,128]
        # print(x.shape)
        # features = torch.flatten(x, 1)
        return {
            'features': x
        }

def unet(args):
     return UNet(n_channels=270,n_classes=55)

# model = UNet(n_channels=270,n_classes=7)
# x = torch.from_numpy(np.random.randint(0, 255, (10, 270, 1000))).float() # 52, 192
# y = model(x)




