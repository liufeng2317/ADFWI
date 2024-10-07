
import torch
import torch.nn.functional as F

class Misfit_traveltime(torch.autograd.Function):
    """Implementation of the cross-correlation misfit function
        s = 0.5*\delta \tau**2
        where \delta \tau is the time shift between synthetic and observed data
        
        Luo, Y., & Schuster, G. T. (1991). Wave-equation traveltime inversion.
            Geophysics, 56(5), 645-653.
    """
    @staticmethod
    def forward(ctx,syn:torch.Tensor,obs:torch.Tensor):
        """forward pass of the cross-correlation misfit function"""
        nsrc,nt,nrec    = syn.shape
        padding         = nt -1 
        max_time_lags   = torch.zeros((nsrc,nrec),device=syn.device,dtype=syn.dtype)
        
        # compute the time shift by cross-correlation
        for isrc in range(nsrc):
            for irec in range(nrec):
                # avoid zero traces
                if syn[isrc,:,irec].sum() == 0 or obs[isrc,:,irec].sum() == 0:
                    continue
                cross_corr = F.conv1d(
                    syn[isrc,:,irec].view(1,1,-1),
                    obs[isrc,:,irec].view(1,1,-1)
                ).squeeze().abs()
                max_time_lags[isrc,irec] = torch.argmax(cross_corr) - padding
        loss = 0.5*torch.sum(max_time_lags**2)
        # save necessary values for backward pass
        ctx.save_for_backward(syn,max_time_lags)
        return loss
    
    @staticmethod
    def backward(ctx,grad_output):
        """backward pass of the cross-correlation travetime misfit function
            grad_output: the gradient of last layer
            
            ds/dv = d(delta tau)/dv * delta tau
                  = 1/(dp_syn/dt) * dp_syn/dv * delta tau
        """
        syn,max_time_lags = ctx.saved_tensors
        # compute the gradient
        adj = torch.zeros_like(syn)
        adj[:,1:-1,:] = (syn[:,2:,:] - syn[:,0:-2,:])/2.0
        adj_sum_squared = torch.sum(adj**2,dim=1,keepdim=True)
        
        # avoid division by zero
        adj_sum_squared[adj_sum_squared == 0] = 1.0
        
        adj = adj/adj_sum_squared
        adj = adj * (max_time_lags).unsqueeze(1)*grad_output
        
        # check for NaNs
        if torch.isnan(adj).any():
            raise ValueError("NaNs in the gradient")
        return adj,None