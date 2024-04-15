import numpy as np
from configs.Metric import Metric

def optimized_cov(a,v):
    nan_mean_a = np.nanmean(a, axis=1).reshape((-1,1))
    nan_mean_b = np.nanmean(v, axis=1).reshape((-1,1))
    return np.nansum((a- nan_mean_a)* (v- nan_mean_b), axis=1)

def mean_difference(a,v):
    return np.nanmean(a) - np.nanmean(v)

def optimized_windowed_cov(a, v, beta=Metric.mean_senssibility_factor):
    if a.shape[1] > v.shape[1]:
        _a, _v = v, a 
    else: 
        _a, _v = a, v

    n = _a.shape[1]
    corrs = []

    for k in range(_v.shape[1] - _a.shape[1]):
        __v = _v[:, k: n+k]
        # Compute the covariance 
        augmented_cov = optimized_cov(_a,__v)+ beta* mean_difference(_a,__v)

        corrs.append(augmented_cov)
        
    return np.array(corrs)

def signature_on_batch(x:np.ndarray, ins:list, outs:list, sig_seq_len:int):
    """Compute the signature from a given batch of MTS sequences `x`

    Args:
        x (np.ndarray): the batch
        ins (list): input columns
        outs (list): output label solumns
        sig_seq_len (int): the desired signature length

    Returns:
        np.ndarray: the min, max, mean signature.
    """
    sigs = []
    shift = sig_seq_len//2
    childrens = x[:, shift:-shift]

    for _in in ins:
        for _out in outs:
            c1 = x[:, :, _in]
            c2 = childrens[:, :, _out]
            
            sig = optimized_windowed_cov(c1, c2)

            sigs.append(sig)

    mins = np.min(sigs, axis=-1)
    maxs = np.max(sigs, axis=-1)
    means= np.mean(sigs, axis=-1)

    signatures = np.stack([mins, maxs, means], axis=-1)

    return signatures

def signature_metric(source_sig:np.ndarray, target_sig:np.ndarray):
    # Shape: (n_features, sign_seq_lenght, 3)
    min_source = source_sig[0]
    max_source = source_sig[1]
    mean_source = source_sig[2]

    min_target = target_sig[0]
    max_target = target_sig[1]
    mean_target = target_sig[2]

    mean_differences = np.mean(mean_target- mean_source)
    area_source = np.mean(max_source- min_source)
    area_target = np.mean(max_target- min_target)

    met = np.power(mean_differences, 2) + Metric.noise_senssitivity*np.power(area_target- area_source, 2)

    return met