import numpy as np
from ADFWI.utils.utils import numpy2tensor


def brutal_picker(trace):
    ''' pick the first arrival 
    '''
    threds = 0.001 * np.max(abs(trace), axis=-1)
    pick = [(abs(trace[i,:]) > threds[i]).argmax(axis=-1) for i in range(trace.shape[0])]
    return np.array(pick)

# functions acting on individual traces
def mask(itmin, itmax, nt, length):
    ''' constructs tapered mask that can be applied to trace to
        mute early or late arrivals.
    '''
    mask = np.ones(nt)
    # construct taper
    win = np.sin(np.linspace(0, np.pi, 2*length))
    win = win[0:length]
    if 1 < itmin < itmax < nt:
        mask[0:itmin] = 0.
        mask[itmin:itmax] = win*mask[itmin:itmax]
    elif itmin < 1 <= itmax:
        mask[0:itmax] = win[length-itmax:length]*mask[0:itmax]
    elif itmin < nt < itmax:
        mask[0:itmin] = 0.
        mask[itmin:nt] = win[0:nt-itmin]*mask[itmin:nt]
    elif itmin > nt:
        mask[:] = 0.
    return mask

def mute_arrival(trace, itmin, itmax, mutetype, nt, length):
    ''' applies tapered mask to record section, muting early or late arrivals
    '''
    win = 1 - mask(itmin, itmax, nt, length)
    win = numpy2tensor(win).to(trace.device)
    trace = trace * win
    return trace
    
def apply_mute(mute_late_window, shot, dt):
    ''' apply time window and offset window mute. [nt,nrcv]
    '''
    shot_np = shot.cpu().detach().numpy()  
    # pick up the first arrival for each trace
    pick = brutal_picker(shot_np.T) + np.ceil(mute_late_window/dt) # [nrcv]
    
    # set parameters
    shot_new = shot.clone()  
    length = 100
    nt = shot_np.shape[0]
    # mask the data for each trach
    itrace = 0
    for i in range(shot.shape[-1]):
        # set mute window
        itmin = int(pick[itrace] - length/2)
        itmax = int(itmin + length)
        # apply mute
        shot_new[:,i] = mute_arrival(shot[:,i], itmin, itmax, 'late', nt, length)
        itrace +=1
    return shot_new