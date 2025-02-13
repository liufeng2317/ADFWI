import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CNN(torch.nn.Module):
    """
        Initialize the CNN model with the given parameters.

        Parameters
        ----------
        model_shape : tuple
            The shape of the velocity model, given as (height, width).
        in_channels : list of int, optional
            The input and output channels for each CNN block. Default is [8, 32, 16].
        vmin : float, optional
            The minimum velocity of the output. Default is None.
        vmax : float, optional
            The maximum velocity of the output. Default is None.
        dropout_prob : float, optional
            Probability for dropout in CNN layers. Default is 0.
        random_state_num : int, optional
            The number of random features for the fully connected input layer. Default is 100.
        unit : int, optional
            The scaling factor for the output. Default is 1000.
        device : str, optional
            Device to use for computation, either 'cpu' or 'cuda'. Default is 'cpu'.

    """
    def __init__(self, model_shape,
                 in_channels=[8, 32, 16],
                 vmin=None, vmax=None,
                 dropout_prob=0,
                 random_state_num=100,
                 unit=1000,
                 device="cpu"):
        super(CNN, self).__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.device = device
        self.unit = unit
        
        # Model configuration
        self.layer_num = len(in_channels) - 1
        h_in = math.ceil(model_shape[0] / (2 ** self.layer_num))
        w_in = math.ceil(model_shape[1] / (2 ** self.layer_num))
        self.h_v0 = model_shape[0]
        self.w_v0 = model_shape[1]
        
        # Neural network blocks
        self.in_features = random_state_num

        # Fully connected input layer followed by reshaping
        self.FNN_in = nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=h_in * w_in * in_channels[0], bias=False),
            nn.Unflatten(0, (-1, in_channels[0], h_in, w_in)),
            nn.LeakyReLU(0.1)
        )
        
        # Convolutional blocks
        self.CNN_Blocks = nn.ModuleList()
        for i in range(self.layer_num):
            self.CNN_Blocks.append(
                nn.Sequential(
                    nn.UpsamplingBilinear2d(scale_factor=(2, 2)),
                    nn.Conv2d(in_channels=in_channels[i], out_channels=in_channels[i + 1], kernel_size=4, stride=1, padding="same", bias=False),
                    nn.LeakyReLU(0.1),
                    nn.Dropout(p=dropout_prob)  # Add dropout layer
                )
            )
        
        # Output convolutional layer
        self.CNN_out = nn.Sequential(
            nn.Conv2d(in_channels=in_channels[-1], out_channels=1, kernel_size=4, stride=1, padding="same", bias=False)
        )
        
        # Latent variable initialization
        torch.manual_seed(1234)
        self.random_latent_vector = torch.rand(self.in_features).to(self.device)
    
    def forward(self):
        """
        Forward pass through the network.

        Returns
        -------
        torch.Tensor
            The output of the CNN model after processing, reshaped and scaled with the given vmin and vmax.
        """
        # Neural network generation
        out = self.FNN_in(self.random_latent_vector)
        for i in range(self.layer_num):
            out = self.CNN_Blocks[i](out)
        out = self.CNN_out(out)
        
        # Post-processing
        out = torch.squeeze(out)
        if self.vmin is not None and self.vmax is not None:
            out = ((self.vmax - self.vmin) * torch.tanh(out) + (self.vmax + self.vmin)) / 2
        out = torch.squeeze(out) * self.unit
        h_v, w_v = out.shape
        h_v0, w_v0 = self.h_v0, self.w_v0
        out = out[(h_v - h_v0) // 2:(h_v - h_v0) // 2 + h_v0,
                  (w_v - w_v0) // 2:(w_v - w_v0) // 2 + w_v0]
        return out