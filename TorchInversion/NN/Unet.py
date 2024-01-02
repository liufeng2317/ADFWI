
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

############################################################################
#                   U-Net model implement by RNN-FWI
############################################################################

# double convolution layer
class DoubleConv(nn.Module):
    """
    (convolution => [BN] => ReLU) * 2
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        """
        :param in_channels: 输入通道数 
        :param out_channels: 双卷积后输出的通道数
        :param mid_channels: 中间的通道数，这个主要针对的是最后一个下采样和上采样层
        :return: 
        """
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding="same", bias=False),
            nn.BatchNorm2d(mid_channels),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.1),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding="same", bias=False),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True)
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    

# upsampling layer
class Up(nn.Module):
    """
    Upscaling then double conv
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        """
        :param in_channels: 输入通道数
        :param out_channels:  输出通道数
        :param bilinear: 是否采用双线性插值，默认采用
        """
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



# class UNet(nn.Module):
#     def __init__(self, init_v,vmin=None,vmax=None,in_channels=1, out_channels=1, base_channel=16, bilinear=False,device="cpu"):
#         super(UNet, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.bilinear = bilinear
#         # input 
#         self.inc = (DoubleConv(in_channels, base_channel))
#         # downstream
#         self.down1 = (Down(base_channel  , base_channel*2))
#         self.down2 = (Down(base_channel*2, base_channel*4))
#         self.down3 = (Down(base_channel*4, base_channel*8))
#         factor = 2 if bilinear else 1
#         self.down4 = (Down(base_channel*8, base_channel*16 // factor))
#         # upstream : upsampling + concate
#         self.up1 = (Up(base_channel*16, 8*base_channel // factor, bilinear))
#         self.up2 = (Up(base_channel*8 , 4*base_channel // factor, bilinear))
#         self.up3 = (Up(base_channel*4 , 2*base_channel // factor, bilinear))
#         self.up4 = (Up(base_channel*2 , base_channel, bilinear))
#         # output stream
#         self.outc = (OutConv(base_channel, out_channels))
        
#         #############################################
#         #                   input
#         #############################################
#         # latent variable
#         self.device = device
#         self.vmin = vmin
#         self.vmax = vmax
#         # output setting
#         h0,w0 = init_v.shape
#         torch.manual_seed(1234)
#         self.random_latent_vector = torch.rand(1,1,h0,w0).to(self.device)
#         # self.random_latent_vector = torch.ones(1,1,h0,w0).to(self.device)
#         # init_v = init_v.to(self.device)
#         # self.random_latent_vector*= init_v

#     def forward(self):
#         x = self.random_latent_vector
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
        
#         out = self.outc(x)
#         out = torch.squeeze(out)
#         if self.vmin != None and self.vmax != None:
#             out = ((self.vmax-self.vmin)*torch.tanh(out) + (self.vmax+self.vmin))/2
#         out = out.squeeze()*1000
#         return out

#     def use_checkpointing(self):
#         self.inc = torch.utils.checkpoint(self.inc)
#         self.down1 = torch.utils.checkpoint(self.down1)
#         self.down2 = torch.utils.checkpoint(self.down2)
#         self.down3 = torch.utils.checkpoint(self.down3)
#         self.down4 = torch.utils.checkpoint(self.down4)
#         self.up1 = torch.utils.checkpoint(self.up1)
#         self.up2 = torch.utils.checkpoint(self.up2)
#         self.up3 = torch.utils.checkpoint(self.up3)
#         self.up4 = torch.utils.checkpoint(self.up4)
#         self.outc = torch.utils.checkpoint(self.outc)


class UNet(nn.Module):
    def __init__(self, init_v,vmin=None,vmax=None,in_channels=1, out_channels=1, base_channel=16, bilinear=False,device="cpu"):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        # input 
        self.inc = (DoubleConv(in_channels, base_channel))
        # downstream
        self.down1 = (Down(base_channel  , base_channel*2))
        factor = 2 if bilinear else 1
        self.down2 = (Down(base_channel*2, base_channel*4 // factor))
        # upstream : upsampling + concate
        self.up1 = (Up(base_channel*4, 2*base_channel // factor, bilinear))
        self.up2 = (Up(base_channel*2 , base_channel, bilinear))
        # output stream
        self.outc = (OutConv(base_channel, out_channels))
        
        #############################################
        #                   input
        #############################################
        # latent variable
        self.device = device
        self.vmin = vmin
        self.vmax = vmax
        # output setting
        h0,w0 = init_v.shape
        torch.manual_seed(1234)
        self.random_latent_vector = torch.rand(1,1,h0,w0).to(self.device)

    def forward(self):
        x = self.random_latent_vector
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        
        out = self.outc(x)
        out = torch.squeeze(out)
        if self.vmin != None and self.vmax != None:
            out = ((self.vmax-self.vmin)*torch.tanh(out) + (self.vmax+self.vmin))/2
        out = torch.squeeze(out)*1000
        return out

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)