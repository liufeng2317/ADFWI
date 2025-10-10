import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, 
                 input_shape=(1000, 64), 
                 output_shape=(128, 128),
                 vmin=None,vmax=None,
                 unit = 1000,
                 device="cpu"
                 ):
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.device = device
        self.unit = unit
        
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),  # [B, 1, nt, nrcv]
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # flatten
        example_input = torch.zeros(1, 1, *input_shape)
        with torch.no_grad():
            x = self.encoder(example_input)
            self.encoded_dim = x.numel()

        # decoder (fully connected)
        self.decoder_fc = nn.Sequential(
            nn.Linear(self.encoded_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_shape[1] * output_shape[2])
        )

        self.output_shape = output_shape

    def forward(self, x):
        out = x.unsqueeze(1)  # [B, 1, nt, nrcv]
        out = self.encoder(out)
        out = out.view(out.size(0), -1)
        out = self.decoder_fc(out)
        out = out.view(*self.output_shape)
        
        # post process
        out = torch.squeeze(out)
        if self.vmin != None and self.vmax != None:
            out = ((self.vmax-self.vmin)*torch.tanh(out) + (self.vmax+self.vmin))/2
        out = torch.squeeze(out)*self.unit
        return out
