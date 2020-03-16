#!/usr/bin/env python
# coding: utf-8

import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
import numpy as np
from .Franke import SinglePathwayModel
from .Franke import *
from .Franke_feature import get_trace_features
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from delfi.simulator.BaseSimulator import BaseSimulator
from delfi.summarystats.BaseSummaryStats import BaseSummaryStats
from scipy import stats as spstats
import delfi.distribution as dd
import delfi.generator as dg
from delfi.utils.viz import samples_nd
import delfi.inference as infer
from sklearn.decomposition import PCA


class Ribon(BaseSimulator):

    def __init__(self, I, dt, dim_param, pre_step, Model, ID, seed=None):
        """Ribon simulator

        Parameters
        ----------
        I : array
            Numpy array with the input current
        dt : float
            Timestep
        dim_param : int
            Number of parameter
        pre_step : int
            Number of time points before the stimuli
        seed : int or None
            If set, randomness across runs is disabled
        """

        super().__init__(dim_param=dim_param, seed=seed)
        self.I = I
        self.dt = dt
        self.t = np.arange(0, len(self.I), 1) * self.dt
        self.RibonModel = Model
        self.pre_step = pre_step
        self.ID = ID

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
        """with open(str(self.ID) + '.log', 'a') as f:
            tmp_params = " ".join([str(params[i]) for i in range(len(params))])
            f.write(tmp_params + '\n')"""
        params = np.asarray(params)

        assert params.ndim == 1, 'params.ndim must be 1'

        hh_seed = self.gen_newseed()

        model = self.RibonModel(params, dt=self.dt, single=True)
        states = model.run(self.I)[0][self.pre_step:]

        return {
            'data': states.reshape(-1),
            'time': self.t,
            'dt': self.dt,
            'I': self.I
        }


class RibonStats(BaseSummaryStats):
    """Moment based SummaryStats class for the Ribon model

    Calculates summary statistics
    """

    def __init__(self, t_on, t_off, dt, n_summary, _type='He', seed=None):
        """See SummaryStats.py for docstring"""
        super(RibonStats, self).__init__(seed=seed)
        self.t_on = t_on
        self.t_off = t_off
        self.dt = dt
        self.n_summary = n_summary
        self.type = _type
        assert self.type in ['PCA', 'He', 'Raw'], print(
            "Type for Ribon statistics should be within ['PCA', 'He', 'Raw'], but got {}"
            .format(self.type))
        self.round = 0

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
        stats = list()
        raw_list = list()
        for r in range(len(repetition_list)):
            x = repetition_list[r]
            assert x['data'].shape[0] == 15996, print(x['data'].shape)
            if self.type == 'He':
                sum_stats_vec = get_trace_features(x['data'], self.dt)  #.reshape(1, -1)
                stats.append(sum_stats_vec)
            elif self.type == 'Raw':
                sum_stats_vec = x['data']
                stats.append(sum_stats_vec)

            elif self.type == 'PCA':
                raise NotImplementedError

        return np.asarray(stats)
