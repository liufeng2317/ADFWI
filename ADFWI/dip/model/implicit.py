import torch
import numpy as np

class IRN(torch.nn.Module):
    '''
    An Implicit Representation Network.
    
    neuron:             a list to define number of layers (length of the list) and number of neurons in each layer;
    activation:         'relu', 'tanh' or 'sine',
                        if 'sine' is chosen, then special initilizations are implemented (see SIREN paper).
    outermost_linear:   if True, then not activation function is applied on the last layer.
    '''
    def __init__(self, 
                 neuron=[2, 256, 256, 256, 256, 1], 
                 omega_0=30, 
                 prob=0.2,
                 bias=True, 
                 dropout=False,
                 outermost_linear=False,
                 activation='sine',
                 vmin=None,vmax=None,
                 unit = 1000,
                 mean=2.6718, 
                 std=0.9738,
                 ):
        super(IRN, self).__init__()
        self.mean = mean
        self.std = std
        self.unit = unit
        self.vmin = vmin
        self.vmax = vmax
        
        self.omega_0 = omega_0
        self.neuron = neuron
        self.d_flag = dropout
        self.outermost_linear = outermost_linear
        
        self.linear = torch.nn.ModuleList()
        self.dropout = torch.nn.ModuleList()
        for idx in range(len(neuron) - 1):
            self.linear.append(torch.nn.Linear(neuron[idx], neuron[idx + 1], bias=bias))
            if self.d_flag:
                self.dropout.append(torch.nn.Dropout(prob, inplace=True))
            
        if activation == 'relu':
            self.omega_0 = 1
            self.activation = torch.nn.ReLU()
        elif activation == 'tanh':
            self.omega_0 = 1
            self.activation = torch.nn.Tanh()
        else:
            self.activation = torch.sin
            self.init_weights()
        
    def init_weights(self):
        with torch.no_grad():
            self.linear[0].weight.uniform_(-1 / self.neuron[0], 1 / self.neuron[0])
            for ix in range(1, len(self.linear)):
                self.linear[ix].weight.uniform_(-np.sqrt(6 / self.neuron[ix]) / self.omega_0, 
                                                 np.sqrt(6 / self.neuron[ix]) / self.omega_0)
            
    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        feature = self.activation(self.omega_0 * self.linear[0](coords))
        for ilayer, layer in enumerate(self.linear[1:]):
            if self.d_flag:
                feature = self.dropout[ilayer](feature)
            feature = layer(feature)
            if not self.outermost_linear or ilayer + 2 < len(self.linear):
                feature = self.activation(self.omega_0 * feature)
        feature = feature.squeeze()
        feature = (feature * self.std + self.mean) * self.unit
        return feature
        
        # post process
        # out = torch.squeeze(feature)
        # if self.vmin != None and self.vmax != None:
        #     out = ((self.vmax-self.vmin)*torch.tanh(out) + (self.vmax+self.vmin))/2
        # out = torch.squeeze(out)*self.unit
        # return out