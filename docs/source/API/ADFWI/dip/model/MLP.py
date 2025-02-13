import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MLP(torch.nn.Module):
    """
    A Multi-layer Perceptron (MLP) model for generating velocity models.

    Parameters
    ----------
    model_shape : tuple
        The shape of the velocity model, given as (height, width).
    random_state_num : int, optional
        The number of random features for the fully connected input layer. Default is 100.
    hidden_layer_number : list of int, optional
        A list specifying the number of neurons in each hidden layer. Default is [100, 100].
    vmin : float, optional
        The minimum velocity of the output. Default is None.
    vmax : float, optional
        The maximum velocity of the output. Default is None.
    unit : int, optional
        The scaling factor for the output. Default is 1000.
    device : str, optional
        The device to use for computation, either 'cpu' or 'cuda'. Default is 'cpu'.
    """
    def __init__(self, model_shape,
                 random_state_num=100,
                 hidden_layer_number=[100, 100],
                 vmin=None, vmax=None,
                 unit=1000,
                 device="cpu"):
        super(MLP, self).__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.device = device
        self.model_shape = model_shape
        self.in_features = random_state_num
        self.unit = unit
        
        # Input layer
        self.MLP_in = nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=hidden_layer_number[0], bias=False),
            nn.LeakyReLU(0.1)
        )

        # Hidden layers
        self.MLP_Blocks = nn.ModuleList()
        for i in range(len(hidden_layer_number) - 1):
            self.MLP_Blocks.append(
                nn.Sequential(
                    nn.Linear(in_features=hidden_layer_number[i], out_features=hidden_layer_number[i + 1]),
                    nn.LeakyReLU(0.1)
                )
            )
        
        # Output layer
        self.MLP_out = nn.Sequential(
            nn.Linear(in_features=hidden_layer_number[-1], out_features=model_shape[0] * model_shape[1])
        )
        
        # Latent variable initialization
        torch.manual_seed(1234)
        self.random_latent_vector = torch.rand(self.in_features).to(self.device)

    def forward(self):
        """
        Forward pass through the MLP.

        Returns
        -------
        torch.Tensor
            The output of the MLP model after processing, reshaped and scaled with the given vmin and vmax.
        """
        # Neural network generation
        out = self.MLP_in(self.random_latent_vector)
        for i in range(len(self.MLP_Blocks)):
            out = self.MLP_Blocks[i](out)
        out = self.MLP_out(out).view(self.model_shape[0], self.model_shape[1])

        # Post-processing
        out = torch.squeeze(out)
        if self.vmin is not None and self.vmax is not None:
            out = ((self.vmax - self.vmin) * torch.tanh(out) + (self.vmax + self.vmin)) / 2
        out = torch.squeeze(out) * self.unit
        return out