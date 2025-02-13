import numpy as np
import torch

##########################################################################
#                          numpy <=====> tensor     
##########################################################################

def numpy2tensor(a,dtype=torch.float32):
    """
        transform numpy data into tensor
    """
    if not torch.is_tensor(a):
        return torch.tensor(a,requires_grad=False,dtype=dtype)
        # return torch.from_numpy(a)
    else:
        return a

def tensor2numpy(a):
    """
        transform tensor data into numpy
    """
    if not torch.is_tensor(a):
        return a 
    else:
        return a.detach().numpy()
    
def gpu2cpu(a):
    if torch.is_tensor(a):
        if a.requires_grad:
            if a.device == 'cpu':
                a = a.detach().numpy()
            else:
                a = a.cpu().detach().numpy()
        else:
            if a.device == 'cpu':
                return a.numpy()
            else:
                return a.cpu().numpy()
    return a

##########################################################################
#                          list <=====> numpy     
##########################################################################
def list2numpy(a):
    """
        transform numpy data into tensor
    """
    if isinstance(a,list):
        return np.array(a)
    else:
        return a

def numpy2list(a):
    """
        transform numpy data into tensor
    """
    if not isinstance(a,list):
        return a.tolist()
    else:
        return a