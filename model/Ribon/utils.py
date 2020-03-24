#coding:UTF-8

import numpy as np
from scipy.special import erf
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

from .configures import *


def get_interp_func(values, min_x, max_x):
    x = np.linspace(min_x, max_x, len(values))
    func = interp1d(x, values, kind='cubic')

    def f(x):
        # print(x, min_x, max_x)
        if x == min_x:
            return values[0]
        elif x == max_x:
            return values[-1]
        else:
            return func(x)

    return np.vectorize(f)


def get_interp_bins(func, min_x, max_x, n_bins):
    x = np.linspace(min_x, max_x, n_bins)
    values = np.vectorize(func)(x)
    return values


def get_linear_func(alphas, delta, max_tau, step=DT):
    max_tau = float(max_tau)
    delta = max(0, delta)
    assert max_tau > 0

    def func(x):
        x += delta
        if x > max_tau or x < 0:
            return 0
        f = 0
        for i, alpha in enumerate(alphas):
            fi = np.sin(np.pi * (i + 1) * (2 * x / max_tau - (x / max_tau)**2))
            f += alpha * fi
        return f

    return np.vectorize(func)


def get_erf_nonlinear_func(a, b1, b2):
    """ for a rising function, a > 1, b2 != 0"""
    K = (a**2 / (1 - b2))

    def func(x, STD):
        return (a**(erf(STD * x + b1) + 1)) / K + b2

    return np.vectorize(func)


def fit_tau(trace, i_begin, dt=DT):
    trace = np.array(trace)
    trace -= trace[-1]
    trace = trace[i_begin:]
    trace = trace / trace[0]
    func = lambda x, TAU: np.exp(-x / TAU)
    time = np.arange(len(trace)) * dt
    popt, pcov = curve_fit(func, time, trace)
    tau = popt[0]
    return tau


def normalize_trace(trace, resting=0):
    trace = np.copy(trace) - resting
    if abs(max(trace)) < abs(min(trace)):
        trace *= -1
    if max(trace) == 0:
        return trace
    trace /= max(trace)
    return trace


def get_func_from_parameters(func_type, parameters):
    if func_type == "None":
        func = lambda x: x
    elif func_type == "constant":
        value = parameters[0]
        func = lambda x: value
    elif func_type == "sigmoid":
        half = parameters[0]
        slope = float(parameters[1])
        m = parameters[2]
        func = lambda x: m / (1. + np.exp(-(x - half) / slope))
    elif func_type == "ahill":
        k = parameters[0]
        m = parameters[1]
        func = lambda x: m / (1. + k / x)
    elif func_type == "rhill":
        k = parameters[0]
        m = parameters[1]
        func = lambda x: m / (1. + x / k)
    elif func_type == "rlinear":
        k = parameters[0]
        func = lambda x: k / max(x, 0.000001)
    else:
        raise Exception("UNKNOWN func types for %s" % func_type)
    return np.vectorize(func)


def sliding_window(X, Y, resting_input, resting_response, run_step, buffer_step,
                   pre_step):
    run_begin = pre_step + buffer_step
    all_time = X.shape[1] - X.shape[1] % run_step  # cut tail
    input_traces = [
        np.zeros((X.shape[0], run_begin + run_step))
        for i in range(0, all_time, run_step)
    ]
    output_traces = [
        np.zeros(run_begin + run_step) for i in range(0, all_time, run_step)
    ]
    for i in input_traces:
        i[:] = resting_input
    for i in output_traces:
        i[:] = resting_response
    for i, index_begin in enumerate(range(0, all_time, run_step)):
        input_traces[i][:, run_begin:] = X[:,
                                           index_begin:index_begin + run_step]
        output_traces[i][run_begin:] = Y[index_begin:index_begin + run_step]
        if i > 0:
            # add buffer to make sure the states at begin is from the trace
            buffer_from_X = max(index_begin - buffer_step, 0)
            buffer_length = index_begin - buffer_from_X
            input_traces[
                i][:, run_begin -
                   buffer_length:run_begin] = X[:, index_begin -
                                                buffer_length:index_begin]
    return input_traces, output_traces


def get_peaks_index(trace):
    min_prev = trace[1:-1] - trace[:-2]
    min_next = trace[1:-1] - trace[2:]
    index = set(np.where(min_prev > 0)[0].tolist())
    index = index.intersection(set(np.where(min_next > 0)[0].tolist()))
    index = sorted(list(index))
    return np.array(index) + 1
