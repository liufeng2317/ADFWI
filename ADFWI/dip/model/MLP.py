import torch
import torch.nn as nn
import torch.nn.functional as F
import math

############################################################################
#  MLP improves fitting speed, but no very good DIP results were found
############################################################################

class MLP(torch.nn.Module):
    def __init__(self,model_shape,
                 random_state_num = 100,
                 hidden_layer_number = [1000,1000],
                 vmin=None,vmax=None,
                 device="cpu"):
        super(MLP,self).__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.device = device
        self.model_shape = model_shape
        self.in_features = random_state_num
        self.MLP_in = nn.Sequential(
            nn.Linear(in_features=self.in_features,out_features=hidden_layer_number[0],bias=False),
            nn.LeakyReLU(0.1)
        )
        self.MLP_Blocks = nn.ModuleList()
        for i in range(len(hidden_layer_number)-1):
            self.MLP_Blocks.append(
                nn.Sequential(
                    nn.Linear(in_features=hidden_layer_number[i],out_features=hidden_layer_number[i+1]),
                    nn.LeakyReLU(0.1)
                )
            )
            
        self.MLP_out = nn.Sequential(
            nn.Linear(in_features=hidden_layer_number[-1],out_features=model_shape[0]*model_shape[1])
        )
        
        # latent variable
        torch.manual_seed(1234)
        self.random_latent_vector = torch.rand(self.in_features).to(self.device)
        
    def forward(self):
        # neural network generation
        out = self.MLP_in(self.random_latent_vector)
        for i in range(len(self.MLP_Blocks)):
            out = self.MLP_Blocks[i](out)
        out = self.MLP_out(out).view(self.model_shape[0],self.model_shape[1])
        # post process
        out = torch.squeeze(out)
        if self.vmin != None and self.vmax != None:
            out = ((self.vmax-self.vmin)*torch.tanh(out) + (self.vmax+self.vmin))/2
        out = torch.squeeze(out)*1000
        return out