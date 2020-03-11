"""
get feature of response trace in (Franke et al., 2017, nature)
"""
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from .configures import *


def get_trace_features(trace, dt=DT):
    """
    #trace: response trace
    #dt: time step of trace
    --------
    return: value of #218 features
    """
    trace -= trace[0]
    features = np.concatenate(
        (get_step_features(trace, dt), get_freq_features(trace, dt),
         get_contrast_features(trace, dt)))
    return features


def get_feature_tau(trace, dt=DT):
    func = lambda x, FLU, TAU: FLU * (1 - np.exp(-x / TAU))
    t = np.arange(len(trace)) * dt
    try:
        popt, pcov = curve_fit(func, t, trace)
        tau = popt[-1]
    except:
        tau = -1
        # print("Error in tau")
    return tau


def get_feature_steplike(response, dt=DT):
    mean = response.mean()
    max = response.max()
    min = response.min()
    features = [mean, response[int(response.shape[0] / 2):].mean()]
    if (max - mean) > (mean - min):
        # has a peak
        features.append(max)
        index = np.where(response == max)[0][0]
        features.append(index * dt)
        trace = response[index] - response[index:]
    else:
        # has a boltten
        features.append(min)
        index = np.where(response == min)[0][0]
        features.append(index * dt)
        trace = response[index:] - response[index]
    try:
        features.append(get_feature_tau(trace, dt))
    except:
        features.append(0.)
    return features


def get_step_features(trace, dt=DT):
    # steps in whole trace
    steps = [
        trace[int(2003):int(4950 / dt)], trace[int(4950 / dt):int(7907 / dt)],
        trace[int(7907 / dt):int(9998 / dt)],
        trace[int(17575 / dt):int(19730 / dt)],
        trace[int(27250 / dt):int(29100 / dt)], trace[int(29100 / dt):]
    ]
    features = []
    for step in steps:
        features += get_feature_steplike(step, dt)
    if len(features) < 30:
        features += [0.] * (30 - len(features))
    elif len(features) > 30:
        features = features[:30]
    return features


def get_freq_features(trace, dt=DT):
    response = trace[int(10000 / dt):int(17575 / dt)]
    peaks = find_peaks(response, distance=int(90 / dt), height=0)[0]
    min_index = np.where(response == response.min())[0][0]
    bottens = find_peaks(-response, distance=int(90 / dt))[0]
    discard_bottens = np.where(bottens <= min_index)[0]
    if len(discard_bottens) > 0:
        bottens = bottens[discard_bottens[-1] + 1:]
    points = np.concatenate((peaks, bottens))
    if len(points) > 63:
        points = points[-63:]
    if len(points) < 63:
        features = np.concatenate((
            points * dt,
            [0.] * (63 - len(points)),
            response[points],
            [0.] * (63 - len(points)),
        ))
    else:
        features = np.concatenate((
            points * dt,
            response[points],
        ))
    return features


def get_contrast_features(trace, dt=DT):
    response = trace[int(19730 / dt):int(27250 / dt)]
    peaks = find_peaks(response, distance=int(200 / dt), height=response[0])[0]
    bottens = find_peaks(-response, distance=int(200 / dt))[0]
    discard_bottens = np.where(bottens <= int(200 / dt))[0]
    if len(discard_bottens) > 0:
        bottens = bottens[discard_bottens[-1] + 1:]
    points = np.concatenate((peaks, bottens))
    if len(points) > 31:
        points = points[-31:]
    if len(points) < 31:
        features = np.concatenate((
            points * dt,
            [0.] * (31 - len(points)),
            response[points],
            [0.] * (31 - len(points)),
        ))
    else:
        features = np.concatenate((
            points * dt,
            response[points],
        ))
    return features
