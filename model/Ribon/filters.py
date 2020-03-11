#coding:UTF-8
"""
Filters for simulation
"""

import numpy as np

from .utils import get_linear_func
from .configures import *


class CascadeFilter:

    def __init__(self, *filters):
        self.filters = filters

    def filter(self, trace):
        for f in self.filters:
            trace = f.filter(trace)
        return trace


class Container:

    def __init__(self, variables, step_func):
        self.variables = variables
        self.step_func = step_func

    def step(self, **parameters):
        self.variables = self.step_func(self.variables, parameters)

    def get_variable(self, name):
        return self.variables[name]

    def set_variable(self, name, value):
        self.variables[name] = value


class ContainerFilter:

    def __init__(self, tau_func, dt=DT):
        self.dt = dt
        self.variables = {
            "v": 0,
        }

        def step(variables, parameters):
            d = -1 * self.variables['v'] + parameters["s"]
            variables["v"] += self.dt * d / max(
                float(tau_func(parameters["s"])), self.dt)
            return variables

        self.container = Container(self.variables, step)

    def f(self, s):
        self.container.step(s=s)
        return self.variables['v']

    def init(self, value):
        self.variables["v"] = value

    def filter(self, trace):
        return np.array([self.f(float(i)) for i in trace])


class ContainerSelfFilter(ContainerFilter):

    def __init__(self, tau_func, dt=DT):
        self.dt = dt
        self.variables = {
            "v": 0,
        }

        def step(variables, parameters):
            d = -1 * self.variables['v'] + parameters["s"]
            variables["v"] += self.dt * d / max(
                float(tau_func(parameters["s"], self)), self.dt)
            return variables

        self.container = Container(self.variables, step)


class AverageFilter:

    def __init__(self, n_window):
        self.n_window = n_window

    def filter(self, trace):
        trace = np.array(trace)
        averaged = np.zeros(trace.shape)
        for i in range(trace.shape[0]):
            d = trace[max(0, int(i - self.n_window / 2)):int(i +
                                                             self.n_window / 2)]
            averaged[i] = d.mean()
        return averaged


class FunctionFilter:

    def __init__(self, func):
        self.func = func

    def filter(self, trace):
        return np.array([self.func(i) for i in trace])


class ICaFilter(FunctionFilter):
    """ from voltage into ica """

    def __init__(self, v50=-35, k=4.8, Imax=1, base=0.05):
        func = lambda x: Imax / (1 + np.exp((v50 - x) / k)) + base
        FunctionFilter.__init__(self, func)


class TemporalFilter:

    def __init__(self, weights, shifted_step=0):
        """ 
        last weight in weights is the latest input 
        """
        weights = np.array(weights)
        self.weights = weights
        self.shifted_step = shifted_step
        self.buffer = np.zeros(self.weights.shape)

    def filter(self, trace):
        trace = np.array(trace)
        assert len(trace.shape) == 1, "Need 1-D Temporal Stimulus"
        trace = np.array(trace)

        def f(x):
            self.buffer[:-1] = self.buffer[1:]
            self.buffer[-1] = x
            return (self.buffer * self.weights).sum()

        return np.vectorize(f)(trace)  #[:-self.shifted_step]


class FunctionTemporalFilter(TemporalFilter):

    def __init__(self, func, step, dt=DT, normalize_weight=False):
        step = int(step / dt)
        weights = np.zeros(step)
        for i in range(step):
            weights[i] = func(i * dt)
        if normalize_weight:
            weights /= abs(weights.sum())
        TemporalFilter.__init__(self, weights[::-1])


class MiddleFilter(FunctionTemporalFilter):
    # K2
    def __init__(self, tau, c, step, dt=DT, keep_head=True):
        tau = float(tau)
        func = lambda x: np.exp(-x / tau) / tau - np.exp(-x /
                                                         (c * tau)) / (c * tau)
        FunctionTemporalFilter.__init__(self, func, step, dt, keep_head)


class SlowFilter(FunctionTemporalFilter):

    def __init__(self, tau, step, dt=DT):
        func = lambda x: np.exp(-float(x) / tau) / tau
        FunctionTemporalFilter.__init__(self,
                                        func,
                                        step,
                                        dt,
                                        normalize_weight=True)


class ConvolutionKernelFilter(FunctionTemporalFilter):
    """
    Temporal from 
    Approximate Bayesian Inference for a Mechanistic
    Model of Vesicle Release at a Ribbon Synapse
    """

    def __init__(self,
                 tau_r=0.05e3,
                 tau_d=0.05e3,
                 tau_phase=150,
                 theta=-np.pi / 7,
                 r=1.,
                 step=250,
                 dt=DT):

        def func(x):
            y = -(x / (r * tau_r)**3) / (1 + x / (r * tau_r))
            y *= np.exp(-(x / (r * tau_d))**2)
            y *= np.cos(2 * np.pi * x / (r * tau_phase) +
                        theta)  # fixed from paper
            return y

        FunctionTemporalFilter.__init__(self,
                                        func,
                                        step,
                                        dt,
                                        normalize_weight=True)


class EPSCFilter:

    def __init__(self, mepsc):
        self.mepsc = mepsc

    def filter(self, trace):
        EPSC = np.zeros(len(trace))
        t_mepsc = self.mepsc.shape[0]
        t_epsc = EPSC.shape[0]
        for i, release in enumerate(trace):
            if t_mepsc + i > t_epsc:
                EPSC[i:i +
                     t_mepsc] += release * self.mepsc[:t_epsc - i - t_mepsc]
            else:
                EPSC[i:i + t_mepsc] += release * self.mepsc
        return EPSC


class TauEPSCFilter(EPSCFilter):

    def __init__(self, A=0.002, raise_tau=0, decay_tau=4, K=4, dt=DT):
        if raise_tau == 0:
            weights = [A]
        else:
            weights = [
                A * (1 - np.exp(-dt * i / raise_tau))
                for i in range(int(raise_tau * K / dt))
            ]
        weights += [
            weights[-1] * np.exp(-dt * i / decay_tau)
            for i in range(int(decay_tau * K / dt))
        ]
        weights = np.array(weights)
        EPSCFilter.__init__(self, weights)


class LinearFilter(TemporalFilter):

    def __init__(self, alphas, delta, max_tau=2000, dt=DT):
        self.delta = delta
        self.max_tau = max_tau
        self.alphas = alphas
        self.func = get_linear_func(alphas, delta, max_tau, step=dt)
        max_t = self.max_tau - self.delta
        n_step = int(max_t / dt)
        x = np.arange(0, n_step) * float(dt)
        weights = self.func(x[::-1])
        weights /= abs(weights.sum())
        TemporalFilter.__init__(self, weights)
