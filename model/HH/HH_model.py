#!/usr/bin/env python
# coding: utf-8

# # Simulation and density estimation of HH model

import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from delfi.simulator.BaseSimulator import BaseSimulator
import delfi.distribution as dd
from delfi.summarystats.BaseSummaryStats import BaseSummaryStats
from scipy import stats as spstats
import delfi.generator as dg
import sys
from delfi.utils.viz import samples_nd
import delfi.inference as infer


def syn_current(duration=120, dt=0.01, t_on=10, curr_level=5e-4, seed=None):
    t_offset = 0.
    duration = duration
    t_off = duration - t_on
    t = np.arange(0, duration + dt, dt)

    # external current
    A_soma = np.pi * ((70. * 1e-4)**2)  # cm2
    I = np.zeros_like(t)
    I[int(np.round(t_on /
                   dt)):int(np.round(t_off /
                                     dt))] = curr_level / A_soma  # muA/cm2

    return I, t_on, t_off, dt, t, A_soma


def ou_stimulus(m, s, time_len, dt, tau):
    assert len(m.shape) == 2

    stimulus_arr = np.zeros(list(m.shape)+[time_len])

    for t in range(time_len - 1):
        stimulus_arr[..., t+1] = stimulus_arr[..., t] - stimulus_arr[..., t] / tau * dt + m * dt + s * np.random.normal(0, 1) * np.sqrt(dt)
    
    return stimulus_arr

def chirp_stimulus(I_C, I_0, time_len, dt):
    L = time_len // 3 * 2
    t = np.arange(0, L) * dt
    sin_curve = I_C * t * np.sin(0.1*t)
    sin_curve = np.tile(sin_curve, I_0.shape) + I_0

    stimulus_arr = np.zeros(list(I_0.shape[:-1])+[time_len])
    stimulus_arr[..., time_len // 6:time_len // 6+L] = I_0 + sin_curve
    stimulus_arr[..., :time_len // 6] = -1.0

    return stimulus_arr

def HHsimulator(V0, params, dt, t, I, seed=None):
    """Simulates the Hodgkin-Huxley model for a specified time duration and current

        Parameters
        ---------- 
        V0 : float
            Voltage at first time step
        params : np.array, 1d of length dim_param
            Parameter vector
        dt : float
            Timestep
        t : array
            Numpy array with the time steps
        I : array
            Numpy array with the input current
        seed : int
        """

    gbar_Na = params[0, 0]  # mS/cm2
    gbar_Na.astype(float)
    gbar_K = params[0, 1]  # mS/cm2
    gbar_K.astype(float)
    gbar_leak = params[0, 2]  # mS/cm2
    gbar_leak.astype(float)
    gbar_M = params[0, 3]  # mS/cm2
    gbar_M.astype(float)
    tau_max = params[0, 4]  # ms
    tau_max.astype(float)
    Vt = params[0, 6]  # mV
    Vt.astype(float)
    nois_fact = params[0, 6]  # uA/cm2
    nois_fact.astype(float)
    E_leak = params[0, 7]  # mV
    E_leak.astype(float)

    # fixed parameters
    C = 1.  # uF/cm2
    E_Na = 53  # mV
    E_K = -107  # mV

    tstep = float(dt)

    if seed is not None:
        rng = np.random.RandomState(seed=seed)
    else:
        rng = np.random.RandomState()

    ####################################
    # kinetics
    def efun(z):
        if np.abs(z) < 1e-4:
            return 1 - z / 2
        else:
            return z / (np.exp(z) - 1)

    def alpha_m(x):
        v1 = x - Vt - 13.
        return 0.32 * efun(-0.25 * v1) / 0.25

    def beta_m(x):
        v1 = x - Vt - 40
        return 0.28 * efun(0.2 * v1) / 0.2

    def alpha_h(x):
        v1 = x - Vt - 17.
        return 0.128 * np.exp(-v1 / 18.)

    def beta_h(x):
        v1 = x - Vt - 40.
        return 4.0 / (1 + np.exp(-0.2 * v1))

    def alpha_n(x):
        v1 = x - Vt - 15.
        return 0.032 * efun(-0.2 * v1) / 0.2

    def beta_n(x):
        v1 = x - Vt - 10.
        return 0.5 * np.exp(-v1 / 40)

    # steady-states and time constants
    def tau_n(x):
        return 1 / (alpha_n(x) + beta_n(x))

    def n_inf(x):
        return alpha_n(x) / (alpha_n(x) + beta_n(x))

    def tau_m(x):
        return 1 / (alpha_m(x) + beta_m(x))

    def m_inf(x):
        return alpha_m(x) / (alpha_m(x) + beta_m(x))

    def tau_h(x):
        return 1 / (alpha_h(x) + beta_h(x))

    def h_inf(x):
        return alpha_h(x) / (alpha_h(x) + beta_h(x))

    # slow non-inactivating K+
    def p_inf(x):
        v1 = x + 35.
        return 1.0 / (1. + np.exp(-0.1 * v1))

    def tau_p(x):
        v1 = x + 35.
        return tau_max / (3.3 * np.exp(0.05 * v1) + np.exp(-0.05 * v1))

    ####################################
    # simulation from initial point
    V = np.zeros_like(t)  # voltage
    n = np.zeros_like(t)
    m = np.zeros_like(t)
    h = np.zeros_like(t)
    p = np.zeros_like(t)

    V[0] = float(V0)
    n[0] = n_inf(V[0])
    m[0] = m_inf(V[0])
    h[0] = h_inf(V[0])
    p[0] = p_inf(V[0])

    for i in range(1, t.shape[0]):
        tau_V_inv = ((m[i - 1]**3) * gbar_Na * h[i - 1] +
                     (n[i - 1]**4) * gbar_K + gbar_leak + gbar_M * p[i - 1]) / C
        V_inf = ((m[i - 1]**3) * gbar_Na * h[i - 1] * E_Na +
                 (n[i - 1]**4) * gbar_K * E_K + gbar_leak * E_leak +
                 gbar_M * p[i - 1] * E_K + I[i - 1] + nois_fact * rng.randn() /
                 (tstep**0.5)) / (tau_V_inv * C)
        V[i] = V_inf + (V[i - 1] - V_inf) * np.exp(-tstep * tau_V_inv)
        n[i] = n_inf(
            V[i]) + (n[i - 1] - n_inf(V[i])) * np.exp(-tstep / tau_n(V[i]))
        m[i] = m_inf(
            V[i]) + (m[i - 1] - m_inf(V[i])) * np.exp(-tstep / tau_m(V[i]))
        h[i] = h_inf(
            V[i]) + (h[i - 1] - h_inf(V[i])) * np.exp(-tstep / tau_h(V[i]))
        p[i] = p_inf(
            V[i]) + (p[i - 1] - p_inf(V[i])) * np.exp(-tstep / tau_p(V[i]))

    return np.array(V).reshape(-1, 1)


class HodgkinHuxley(BaseSimulator):

    def __init__(self, I, dt, V0, dim_param, seed=None):
        """Hodgkin-Huxley simulator

        Parameters
        ----------
        I : array
            Numpy array with the input current
        dt : float
            Timestep
        V0 : float
            Voltage at first time step
        seed : int or None
            If set, randomness across runs is disabled
        """

        super().__init__(dim_param=dim_param, seed=seed)
        self.I = I
        self.dt = dt
        self.t = np.arange(0, len(self.I), 1) * self.dt
        self.HHsimulator = HHsimulator
        self.init = V0

    def gen_single(self, params):
        """Forward model for simulator for single parameter set

        Parameters
        ----------
        params : list or np.array, 1d of length dim_param
            Parameter vector

        Returns
        -------
        dict : dictionary with data
            The dictionary must contain a key data that contains the results of
            the forward run. Additional entries can be present.
        """
        params = np.asarray(params)

        assert params.ndim == 1, 'params.ndim must be 1'

        hh_seed = self.gen_newseed()

        states_list = list()
        for i in range(self.I.shape[0]):
            states_list.append(self.HHsimulator(self.init,
                                    params.reshape(1, -1),
                                    self.dt,
                                    self.t,
                                    self.I[i],
                                    seed=hh_seed))
        states = np.concatenate(states_list, axis=0)

        return {
            'data': states.reshape(-1),
        }


# We want to fit a set of summary statistics of the observed data: number of spikes, mean resting potential, standard deviation of the resting potential, and the first 4 voltage moments, mean, standard deviation, skewness and kurtosis. In order to do accomplish that, we define a summary statistics class which computes those quantities:


class HodgkinHuxleyStats(BaseSummaryStats):
    """Moment based SummaryStats class for the Hodgkin-Huxley model

    Calculates summary statistics
    """

    def __init__(self, t_on, t_off, n_mom=4, n_summary=7, seed=None):
        """See SummaryStats.py for docstring"""
        super(HodgkinHuxleyStats, self).__init__(seed=seed)
        self.t_on = t_on
        self.t_off = t_off
        self.n_mom = n_mom
        self.n_summary = np.minimum(n_summary, n_mom + 3)

    def calc(self, repetition_list):
        """Calculate summary statistics

        Parameters
        ----------
        repetition_list : list of dictionaries, one per repetition
            data list, returned by `gen` method of Simulator instance

        Returns
        -------
        np.array, 2d with n_reps x n_summary
        """
        stats = []
        for r in range(len(repetition_list)):
            x = repetition_list[r]

            N = x['data'].shape[0]
            t = x['time']
            dt = x['dt']
            t_on = self.t_on
            t_off = self.t_off

            # initialise array of spike counts
            v = np.array(x['data'])

            # put everything to -10 that is below -10 or has negative slope
            ind = np.where(v < -10)
            v[ind] = -10
            ind = np.where(np.diff(v) < 0)
            v[ind] = -10

            # remaining negative slopes are at spike peaks
            ind = np.where(np.diff(v) < 0)
            spike_times = np.array(t)[ind]
            spike_times_stim = spike_times[(spike_times > t_on) &
                                           (spike_times < t_off)]

            # number of spikes
            if spike_times_stim.shape[0] > 0:
                spike_times_stim = spike_times_stim[
                    np.append(1, np.diff(spike_times_stim)) > 0.5]

            # resting potential and std
            rest_pot = np.mean(x['data'][t < t_on])
            rest_pot_std = np.std(x['data'][int(.9 * t_on / dt):int(t_on / dt)])

            # moments
            std_pw = np.power(np.std(x['data'][(t > t_on) & (t < t_off)]),
                              np.linspace(3, self.n_mom, self.n_mom - 2))
            std_pw = np.concatenate((np.ones(1), std_pw))
            moments = spstats.moment(x['data'][(t > t_on) & (t < t_off)],
                                     np.linspace(2, self.n_mom,
                                                 self.n_mom - 1)) / std_pw

            # concatenation of summary statistics
            sum_stats_vec = np.concatenate(
                (np.array([spike_times_stim.shape[0]]),
                 np.array([
                     rest_pot, rest_pot_std,
                     np.mean(x['data'][(t > t_on) & (t < t_off)])
                 ]), moments))
            sum_stats_vec = sum_stats_vec[0:self.n_summary]

            stats.append(sum_stats_vec)

        return np.asarray(stats)
