"""
get feature of response trace in (Franke et al., 2017, nature)
"""
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from Franke import get_data_pair_lchirp

CLUSTER_ID = 1

MAX_TAU = 2000

DT = 0.1
_, _, TARGET = get_data_pair_lchirp(CLUSTER_ID, dt=DT)


def get_trace_discrepancy(trace, dt):
    """
    #trace: response trace
    #dt: time step of trace
    -------
    return: vector of discrepancy values in trace (0~1)
    notice: please use the function `set_target_features` first
    """
    trace_features = get_trace_features(trace, dt=dt)
    return np.abs(trace_features - TARGET_features)


def set_target_features(dt):
    global TARGET_features
    TARGET_features = get_trace_features(TARGET[::int(dt / DT)], dt=dt)


def get_trace_features(trace, dt=DT):
    """
    #trace: response trace
    #dt: time step of trace
    """
    n_pre = int(2000 / dt)
    trace -= trace[:n_pre].mean()
    target = TARGET[::int(dt / DT)]
    features = np.concatenate(
        ([min(np.abs(trace - target).sum() / np.abs(target).sum(),
              1)], get_step_features(trace, dt), get_freq_features(trace, dt),
         get_contrast_features(trace, dt)))
    return features


def get_feature_tau(trace, dt=DT):
    func = lambda x, FLU, TAU: FLU * (1 - np.exp(-x / TAU))
    t = np.arange(len(trace)) * dt
    popt, pcov = curve_fit(func, t, trace)
    tau = popt[-1]
    return tau


@np.vectorize
def normalize_resp(value):
    return (value + 1) / 2.


def get_feature_steplike(response, dt=DT):
    mean = response.mean()
    max = response.max()
    min = response.min()
    features = [mean, response[int(response.shape[0] / 2):].mean()]
    features = normalize_resp(features).tolist()
    if (max - mean) > (mean - min):
        # has a peak
        features.append(normalize_resp(max))
        index = np.where(response == max)[0][0]
        peak_time = float(index) / len(response)
        features.append(peak_time)
        trace = response[index] - response[index:]
    else:
        # has a boltten
        features.append(normalize_resp(min))
        index = np.where(response == min)[0][0]
        botten_time = float(index) / len(response)
        features.append(botten_time)
        trace = response[index:] - response[index]
    try:
        tau = get_feature_tau(trace, dt)
        tau /= MAX_TAU
        tau = min(1, tau)
        features.append(tau)
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
            points / float(len(response)),
            [0.] * (63 - len(points)),
            normalize_resp(response[points]),
            [0.] * (63 - len(points)),
        ))
    else:
        features = np.concatenate((
            points / float(len(response)),
            normalize_resp(response[points]),
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
            points / float(len(response)),
            [0.] * (31 - len(points)),
            normalize_resp(response[points]),
            [0.] * (31 - len(points)),
        ))
    else:
        features = np.concatenate((
            points / float(len(response)),
            normalize_resp(response[points]),
        ))
    return features
