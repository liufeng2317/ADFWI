import numpy as np
from skimage.metrics import structural_similarity as ssim

def MSE(true_v,inv_v):
    nz,nx = true_v.shape
    return np.sum((true_v-inv_v)**2)

def RMSE(true_v,inv_v):
    if len(true_v.shape) != len(inv_v.shape):
        true_v = true_v[np.newaxis,:,:]
        res = np.sqrt(np.sum((true_v-inv_v)**2,axis=(1,2)))
        return res
    else:
        return np.sqrt(np.sum((true_v-inv_v)**2))

def MAPE(true_v,inv_v):
    if len(true_v.shape) != len(inv_v.shape):
        nz,nx = true_v.shape
        true_v = true_v[np.newaxis,:,:]
        res = 100/(nx*nz) * np.sum(np.abs(inv_v-true_v)/true_v,axis=(1,2))
    else:
        nz,nx = true_v.shape[-2:]
        res = 100/(nx*nz) * np.sum(np.abs(inv_v-true_v)/true_v)
    return res

def SSIM(true_v,inv_v,win_size=11):
    if len(true_v.shape) != len(inv_v.shape):
        ssim_res = []
        for i in range(inv_v.shape[0]):
            vmax = np.max([true_v.max(),inv_v[i].max()])
            vmin = np.min([true_v.min(),inv_v[i].min()])
            temp_ssim =  ssim(true_v,inv_v[i],data_range=vmax-vmin,win_size=win_size)
            ssim_res.append(temp_ssim)
        return ssim_res
    else:
        vmax = np.max([true_v.max(),inv_v.max()])
        vmin = np.min([true_v.min(),inv_v.min()])
        return ssim(true_v,inv_v,data_range=vmax-vmin,win_size=win_size)
    
def SNR(true_v,inv_v):
    if len(true_v.shape) != len(inv_v.shape):
        snr_res = []
        for i in range(inv_v.shape[0]):
            true_norm  = np.sum(true_v**2)
            diff_norm = np.sum((true_v-inv_v[i])**2)
            temp_snr = 10*np.log10(true_norm/diff_norm)
            snr_res.append(temp_snr)
        return snr_res
    else:
        true_norm  = np.sum(true_v**2)
        diff_norm = np.sum((true_v-inv_v)**2)
        temp_snr = 10*np.log10(true_norm/diff_norm)
        return temp_snr