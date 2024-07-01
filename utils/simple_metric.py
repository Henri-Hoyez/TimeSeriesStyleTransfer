import numpy as np


def moving_average(arr, n=3):
    return np.array([np.convolve(arr[:, i], np.ones(n)/n, mode='same') for i in range(arr.shape[-1])]).T

def simple_metric_on_noise(signals:np.ndarray):
    # Here the goal is, knowing the noise domain shift, 
    # extract the trend and the noise from a the signal.
    trends = np.array([moving_average(s) for s in signals])

    residual_noises = np.mean(np.std(trends - signals, axis=1), axis=0)
    return trends, residual_noises