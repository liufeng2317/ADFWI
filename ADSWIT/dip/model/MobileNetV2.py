import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MobileNetV2FWI(nn.Module):
    def __init__(self, model_shape, init_conv_num=4,inverted_residual_blocks=6, vmin=None, vmax=None, random_state_num=100, device="cpu"):
        """
            model_shape (tuple): the shape of velocity model
            inverted_residual_blocks (int): the number of inverted residual blocks, up to 6
            vmin (float): the minimum velocity of output
            vmax (float): the maximum velocity of output
            device (optional): cpu or cuda
        """
        super(MobileNetV2FWI, self).__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.device = device
        self.model_shape = model_shape
        self.init_conv_num = init_conv_num
        # Initial conv layer
        self.init_conv = nn.Sequential(
            nn.Conv2d(init_conv_num, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        # Dynamically create inverted residual blocks
        self.inverted_residual_blocks = nn.ModuleList()
        inp = 32
        channels = [64, 96, 160, 320, 640, 1280]
        for i in range(min(inverted_residual_blocks, 6)):
            outp = channels[i % len(channels)]
            stride = 2 if i % len(channels) == 0 else 1
            self.inverted_residual_blocks.append(InvertedResidual(inp, outp, stride, expand_ratio=6))
            inp = outp

        # Final conv layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(inp, 1280, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )

        # Final layer to map to the desired output channels
        self.output_conv = nn.Conv2d(1280, 1, kernel_size=1, bias=False)
        
        # Resize output to match model_shape
        self.resize = nn.Upsample(size=model_shape, mode='bilinear', align_corners=True)

        # Latent variable
        self.in_features = random_state_num
        torch.manual_seed(1234)
        self.random_latent_vector = torch.rand(self.in_features).to(self.device)

        # Fully connected layer to reshape the latent vector to match MobileNetV2 input size
        self.fc = nn.Linear(self.in_features, self.init_conv_num * (model_shape[0] // 2) * (model_shape[1] // 2))
        
    def forward(self):
        # Generate initial features from the random latent vector
        out = self.fc(self.random_latent_vector)
        out = out.view(1, self.init_conv_num, self.model_shape[0] // 2, self.model_shape[1] // 2)
        
        # Initial conv layer
        out = self.init_conv(out)

        # Pass through inverted residual blocks
        for block in self.inverted_residual_blocks:
            out = block(out)

        # Final conv layer
        out = self.final_conv(out)
        
        # Final convolution to map to single output channel
        out = self.output_conv(out)
        
        # Resize to desired model shape
        out = self.resize(out)
        
        # Post-process the output
        out = torch.squeeze(out)
        if self.vmin is not None and self.vmax is not None:
            out = ((self.vmax - self.vmin) * torch.tanh(out) + (self.vmax + self.vmin)) / 2
        out = out * 1000
        
        return out

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(nn.Conv2d(inp, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        # dw
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.ReLU6(inplace=True))
        # pw-linear
        layers.append(nn.Conv2d(hidden_dim, oup, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(oup))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

# # Example usage:
# model_shape = (128, 128)
# model = SimpleMobileNetV2(model_shape, inverted_residual_blocks=6, vmin=1.5, vmax=4.5, device="cpu")
# output = model()
# print(output.shape)
