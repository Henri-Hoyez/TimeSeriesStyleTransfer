import numpy as np

def moving_average(arr, n=3):
    return np.array([np.convolve(arr[:, i], np.ones(n)/n, mode='same') for i in range(arr.shape[-1])]).T

def simple_metric_on_noise(signals:np.ndarray):
    # Here the goal is, knowing the noise domain shift, 
    # extract the trend and the noise from a the signal.
    trends = np.array([moving_average(s) for s in signals])

    residual_noises = np.mean(np.std(trends - signals, axis=1), axis=0)
    return trends, residual_noises


#### Amplitude shift
def extract_amplitude_from_signals(batch):
    _mins = np.mean(np.min(batch, axis=1), axis=0)
    _maxs = np.mean(np.max(batch, axis=1), axis=0)

    return _maxs- _mins

def simple_amplitude_metric(reference_batch, generated_batch):
    ref_ampls = extract_amplitude_from_signals(reference_batch)
    gen_ampls = extract_amplitude_from_signals(generated_batch)

    return np.mean(np.abs(gen_ampls- ref_ampls))