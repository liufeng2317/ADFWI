import torch
import torch.nn as nn
import torch.nn.functional as F

# Double convolution block
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.1),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.double_conv(x)

# Downscaling block
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

# Upscaling block
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# Output layer
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# General UNet class for any number of layers
class UNet(nn.Module):
    def __init__(self, model_shape, n_layers, base_channel, vmin=None, vmax=None, in_channels=1, out_channels=1, bilinear=False, device="cpu"):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, base_channel)
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        # Create down blocks
        channels = base_channel
        for _ in range(n_layers):
            self.downs.append(Down(channels, channels * 2))
            channels *= 2

        # Create up blocks
        factor = 2 if bilinear else 1
        for _ in range(n_layers):
            self.ups.append(Up(channels, channels // 2 // factor, bilinear))
            channels //= 2

        self.outc = OutConv(channels, out_channels)

        # Random latent variable for input
        self.device = device
        self.vmin = vmin
        self.vmax = vmax
        h0, w0 = model_shape
        torch.manual_seed(1234)
        self.random_latent_vector = torch.rand(1, 1, h0, w0).to(self.device)

    def forward(self):
        x = self.random_latent_vector
        x1 = self.inc(x)

        downs_outputs = [x1]
        for down in self.downs:
            downs_outputs.append(down(downs_outputs[-1]))

        x = downs_outputs[-1]
        for i in range(len(self.ups)):
            x = self.ups[i](x, downs_outputs[-2-i])

        out = self.outc(x)
        out = torch.squeeze(out)
        if self.vmin is not None and self.vmax is not None:
            out = ((self.vmax - self.vmin) * torch.tanh(out) + (self.vmax + self.vmin)) / 2
        out = torch.squeeze(out) * 1000
        return out
