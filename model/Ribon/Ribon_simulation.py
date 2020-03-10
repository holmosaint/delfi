#!/usr/bin/env python
# coding: utf-8

import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
import numpy as np
from Franke import SinglePathwayModel
from Franke import *
from Franke_feature import get_trace_features
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from delfi.simulator.BaseSimulator import BaseSimulator
from delfi.summarystats.BaseSummaryStats import BaseSummaryStats
from scipy import stats as spstats
import delfi.distribution as dd
import delfi.generator as dg
from delfi.utils.viz import samples_nd

dt = 2

pre_step, stim, response = get_data_pair_lchirp(7, 20000, DATATYPE=1, dt=dt)
feature = get_trace_features(response, dt=dt)

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
        self.t = np.arange(0, len(self.I), 1)*self.dt
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
        states = model.run(self.I)[0][pre_step:]

        return {'data': states.reshape(-1),
                'time': self.t,
                'dt': self.dt,
                'I': self.I}

class RibonStats(BaseSummaryStats):
    """Moment based SummaryStats class for the Ribon model

    Calculates summary statistics
    """
    def __init__(self, t_on, t_off, dt, n_summary, seed=None):
        """See SummaryStats.py for docstring"""
        super(RibonStats, self).__init__(seed=seed)
        self.t_on = t_on
        self.t_off = t_off
        self.dt = dt
        self.n_summary = n_summary

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
            assert x['data'].shape[0] == 15996, print(x['data'].shape)
            sum_stats_vec = get_trace_features(x['data'], self.dt)#.reshape(1, -1)
            # sum_stats_vec = x['data']
            stats.append(sum_stats_vec)

        return np.asarray(stats)


param_dim = 9

x = [np.array([[0.1, 4]]),  # half of sigmoid
     np.array([[0.1, 3]]), # slope of sigmoid
     np.array([[0.1, 5]]), # maximum of sigmoid
     np.array([[0.1, 50]]), # tau fo temporal filter
     np.array([[0.01, 2]]), # k of gain infi
     np.array([[0.2, 3]]), # m of gain infi
     np.array([[-2, +2]]), # half of gain tau
     np.array([[-1, +1]]), # slope of gain tau
     np.array([[100, 1500]])] # max of gain tau

label_params = ['half of sigmoid',
                'slope of sigmoid',
                'maximum of sigmoid',
                'tau fo temporal filter',
                'k of gain infi', 
                'm of gain infi',
                'half of gain tau',
                'slope of gain tau',
                'max of gain tau']

import sys
import os

_dir = sys.argv[1]
data_file_name = sys.argv[2]

seed_p = np.random.randint(1000)
prior_min = np.concatenate(x, axis=0)[:, 0]
prior_max = np.concatenate(x, axis=0)[:, 1]
prior = dd.Uniform(lower=prior_min, upper=prior_max,seed=seed_p)

# input current, time step
t_on = 0
t_off = len(stim)

# seeds
seed_m = 1

# summary statistics hyperparameters
n_summary = 218

# define model, prior, summary statistics and generator classes
s = RibonStats(t_on=t_on, t_off=t_off, dt=dt, n_summary=n_summary)

n_processes = 20

seeds_m = np.arange(1, n_processes+1, 1)
m = []
for i in range(n_processes):
    m.append(Ribon(stim, dt, param_dim, pre_step, SinglePathwayModel, ID=i, seed=np.random.randint(1000)))
g = dg.MPGenerator(models=m, prior=prior, summary=s, data_file_name=os.path.join(_dir, data_file_name))

seed_inf = 1

pilot_samples = int(1e4)

# training schedule
n_train = 100
n_rounds = 10

# fitting setup
minibatch = 128
epochs = 10
val_frac = 0.05

# network setup
n_hiddens = [100, 100]

# convenience
prior_norm = True

# MAF parameters
density = 'maf'
n_mades = 5         # number of MADES

import delfi.inference as infer

# inference object
res = infer.SNPEC(g,
                obs=feature,
                n_hiddens=n_hiddens,
                seed=seed_inf,
                pilot_samples=pilot_samples,
                n_mades=n_mades,
                prior_norm=prior_norm,
                density=density)

# train
log, _, posterior = res.run(
                    n_train=n_train,
                    n_rounds=n_rounds,
                    minibatch=minibatch,
                    epochs=epochs,
                    silent_fail=False,
                    proposal='prior',
                    val_frac=val_frac,
                    verbose=True,)

fig = plt.figure(figsize=(15,5))

plt.plot(log[0]['loss'],lw=2)
plt.xlabel('iteration')
plt.ylabel('loss')

plt.savefig('loss.png', dpi=400)

prior_min = g.prior.lower
prior_max = g.prior.upper
prior_lims = np.concatenate((prior_min.reshape(-1,1),prior_max.reshape(-1,1)),axis=1)

posterior_samples = posterior[0].gen(10000)
np.save(os.path.join(_dir, 'posterior_samples.npy'), posterior_samples)

###################
# colors
hex2rgb = lambda h: tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

# RGB colors in [0, 255]
col = {}
col['GT']      = hex2rgb('30C05D')
col['SNPE']    = hex2rgb('2E7FE8')
col['SAMPLE1'] = hex2rgb('8D62BC')
col['SAMPLE2'] = hex2rgb('AF99EF')

# convert to RGB colors in [0, 1]
for k, v in col.items():
    col[k] = tuple([i/255 for i in v])

###################
# posterior
fig, axes = samples_nd(posterior_samples,
                       limits=prior_lims,
                       ticks=prior_lims,
                       labels=label_params,
                       fig_size=(5,5),
                       diag='kde',
                       upper='kde',
                       hist_diag={'bins': 50},
                       hist_offdiag={'bins': 50},
                       kde_diag={'bins': 50, 'color': col['SNPE']},
                       kde_offdiag={'bins': 50},
                       # points=[true_params],
                       points_offdiag={'markersize': 5},
                       points_colors=[col['GT']],
                       title='');

plt.savefig(os.path.join(_dir, 'posterior.png'), dpi=400)

fig = plt.figure(figsize=(7,5))

y_obs = response
t = np.arange(y_obs.shape[0])
duration = np.max(t)

num_samp = 200

# sample from posterior
x_samp = posterior[0].gen(n_samples=num_samp)

# reject samples for which prior is zero
ind = (x_samp > prior_min) & (x_samp < prior_max)
params = x_samp[np.prod(ind,axis=1)==1]

num_samp = min(2, len(params[:,0]))

# simulate and plot samples
V = np.zeros((len(t), num_samp))
for i in range(num_samp):
    x = m[0].gen_single(params[i,:])
    V[:,i] = x
    plt.plot(t, V[:, i], color = col['SAMPLE'+str(i+1)], lw=2, label='sample '+str(num_samp-i))

    # plot observation
    plt.plot(t, y_obs, '--',lw=2, label='observation')
    plt.xlabel('time (ms)')
    plt.ylabel('voltage (mV)')

    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.3, 1), loc='upper right')

    ax.set_xticks([0, duration/2, duration])
    ax.set_yticks([-80, -20, 40]);

plt.savefig(os.path.join(_dir, 'result.png'), dpi=400)
