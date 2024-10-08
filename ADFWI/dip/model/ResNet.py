import torch
import torch.nn as nn
import torch.nn.functional as F

def act(act_fun='LeakyReLU'):
    if isinstance(act_fun, str):
        if act_fun == 'LeakyReLU':
            return nn.LeakyReLU(0.1, inplace=True)
        elif act_fun == 'Swish':
            return nn.SiLU()  # PyTorch equivalent of Swish
        elif act_fun == 'ELU':
            return nn.ELU()
        elif act_fun == 'none':
            return nn.Identity()  # Use Identity instead of empty Sequential
        else:
            raise ValueError(f"Unsupported activation: {act_fun}")
    else:
        return act_fun()

def conv(in_f, out_f, kernel_size=3, stride=1, bias=True, pad='same', downsample_mode='stride'):
    downsampler = None
    if stride != 1 and downsample_mode != 'stride':
        if downsample_mode == 'avg':
            downsampler = nn.AvgPool2d(stride, stride)
        elif downsample_mode == 'max':
            downsampler = nn.MaxPool2d(stride, stride)
        else:
            raise ValueError(f"Unsupported downsample mode: {downsample_mode}")
        stride = 1  # Reset stride after pooling

    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=pad, bias=bias)

    layers = [convolver]
    if downsampler:
        layers.append(downsampler)
    return nn.Sequential(*layers)

def get_block(num_channels, norm_layer, act_fun):
    layers = [
        nn.Conv2d(num_channels, num_channels, 3, 1, 1, bias=False),
        norm_layer(num_channels, affine=True),
        act(act_fun),
        nn.Conv2d(num_channels, num_channels, 3, 1, 1, bias=False),
        norm_layer(num_channels, affine=True),
    ]
    return nn.Sequential(*layers)

class ResidualSequential(nn.Module):
    def __init__(self, *args):
        super(ResidualSequential, self).__init__()
        self.model = nn.Sequential(*args)

    def forward(self, x):
        out = self.model(x)
        if out.size(2) != x.size(2) or out.size(3) != x.size(3):
            x = F.interpolate(x, size=out.size()[2:], mode='bilinear', align_corners=False)
        return out + x

class ResNet(nn.Module):
    def __init__(self, model_shape, vmin=None, vmax=None, num_input_channels=1, num_output_channels=1, 
                 num_blocks=8, num_channels=32, need_residual=True, act_fun='LeakyReLU', 
                 need_sigmoid=True, norm_layer=nn.BatchNorm2d, pad='same', device="cpu"):
        super(ResNet, self).__init__()

        block_class = ResidualSequential if need_residual else nn.Sequential

        layers = [
            conv(num_input_channels, num_channels, 3, stride=1, bias=True, pad=pad),
            act(act_fun)
        ]
        
        for _ in range(num_blocks):
            layers.append(block_class(*get_block(num_channels, norm_layer, act_fun)))
        
        layers += [
            nn.Conv2d(num_channels, num_channels, 3, 1, 1),
            norm_layer(num_channels, affine=True),
            conv(num_channels, num_output_channels, 3, 1, bias=True, pad=pad),
            nn.LeakyReLU(0.1)
        ]
        
        self.model = nn.Sequential(*layers)
        self.device = device
        self.vmin = vmin
        self.vmax = vmax
        
        torch.manual_seed(1234)
        self.random_latent_vector = torch.rand(1, 1, model_shape[0], model_shape[1], device=self.device)

    def forward(self):
        out = self.model(self.random_latent_vector)
        out = torch.squeeze(out)
        if self.vmin is not None and self.vmax is not None:
            out = ((self.vmax - self.vmin) * torch.tanh(out) + (self.vmax + self.vmin)) / 2
        return out * 1000

    def eval(self):
        self.model.eval()