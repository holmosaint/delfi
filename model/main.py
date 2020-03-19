import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import sys
import argparse
import numpy as np
import h5py as h5

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from delfi.simulator.BaseSimulator import BaseSimulator
import delfi.distribution as dd
from delfi.summarystats.BaseSummaryStats import BaseSummaryStats
from scipy import stats as spstats
import delfi.generator as dg
from delfi.utils.viz import samples_nd
import delfi.inference as infer

from HH.HH_model import syn_current, HHsimulator, HodgkinHuxley, HodgkinHuxleyStats
from Ribon.Franke import get_data_pair_lchirp, SinglePathwayModel
from Ribon.Franke_feature import get_trace_features
from Ribon.Ribon_simulation import Ribon, RibonStats


def load_data(pilot_file):
    with h5.File(pilot_file, 'r') as f:
        paras = f.get('param_data')[...]
        stats = f.get('stats_data')[...]
    return paras, stats


def run(args):
    model_name = args.model_name
    n_summary = args.n_summary
    n_processes = args.n_processes
    store_file = args.store_file
    pilot_samples = args.pilot_samples
    pilot_file = args.pilot_file
    n_train = args.n_train
    n_rounds = args.n_rounds
    minibatch = args.batch
    epochs = args.epoch
    val_frac = args.val_frac
    dim_hiddens = args.dim_hiddens
    hidden_layer = args.hidden_layer
    n_hiddens = [dim_hiddens for i in range(hidden_layer)]
    n_mades = args.n_mades  # number of MADES
    result_dir = args.result_dir
    PCA_file = args.PCA_file
    dt = args.dt
    density = args.density
    proposal = args.proposal
    feature_type = args.feature

    if model_name == 'HH':
        prior_min = np.array([0.5, 1e-4, 1e-4, 1e-4, 50, 40, 1e-4, 35])
        prior_max = np.array([80., 15, 0.6, 0.6, 3000, 90., 0.15, 100])
        prior = dd.Uniform(lower=prior_min,
                           upper=prior_max,
                           seed=np.random.randint(1000))

        # true parameters and respective labels
        true_params = np.array([50., 1., 0.03, 0.03, 100, 42, 0.12, 50])
        labels_params = [
            r'$g_{Na}$', r'$g_{K}$', r'$g_{L}$', r'$g_{M}$', r'$\tau_{max}$',
            r'$V_{T}$', r'$\sigma$', r'$E_{L}$'
        ]

        # input current, time step
        I, t_on, t_off, dt, t, A_soma = syn_current()

        # initial voltage
        V0 = -70

        # parameters dimension
        dim_param = 8

        # summary statistics hyperparameters
        n_mom = 4

        s = HodgkinHuxleyStats(t_on=t_on,
                               t_off=t_off,
                               n_mom=n_mom,
                               n_summary=n_summary)

        m = []
        for i in range(n_processes):
            m.append(
                HodgkinHuxley(I,
                              dt,
                              V0=V0,
                              dim_param=dim_param,
                              seed=np.random.randint(1000)))
        g = dg.MPGenerator(data_file_name=store_file,
                           models=m,
                           prior=prior,
                           summary=s)

        # observed data: simulation given true parameters
        obs = m[0].gen_single(true_params)
        obs_stats = s.calc([obs])
        obs = obs['data']

    elif model_name == 'Ribon':
        assert dt == 2, print("dt in Ribon model should be 2 currently")
        """assert n_summary == 218, print(
            "n_summary in Ribon model should be 218 currently")"""

        pre_step, stim, obs = get_data_pair_lchirp(1, 20000, DATATYPE=1, dt=dt)
        obs_stats = get_trace_features(obs, dt=dt)

        dim_param = 9

        true_params = None

        param_range = [
            np.array([[0.1, 4]]),  # half of sigmoid
            np.array([[0.1, 3]]),  # slope of sigmoid
            np.array([[0.1, 5]]),  # maximum of sigmoid
            np.array([[0.1, 50]]),  # tau fo temporal filter
            np.array([[0.01, 2]]),  # k of gain infi
            np.array([[0.2, 3]]),  # m of gain infi
            np.array([[-2, +2]]),  # half of gain tau
            np.array([[-1, +1]]),  # slope of gain tau
            np.array([[100, 1500]])
        ]  # max of gain tau

        labels_params = [
            r'${sigmoid}_{HALF}$', r'${sigmoid}_{SLOPE}$', r'${sigmoid}_{MAX}$',
            r'$\tau_{TF}$', r'$g\_{INF}_k$', r'$g\_{INF}_k$',
            r'$g\_\tau_{HALF}$', r'$g\_\tau_{SLOPE}$', r'$g\_\tau_{MAX}$'
        ]

        prior_min = np.concatenate(param_range, axis=0)[:, 0]
        prior_max = np.concatenate(param_range, axis=0)[:, 1]
        prior = dd.Uniform(lower=prior_min,
                           upper=prior_max,
                           seed=np.random.randint(1000))

        # input current, time step
        t_on = 0
        t_off = len(stim)

        # define model, prior, summary statistics and generator classes
        s = RibonStats(t_on=t_on,
                       t_off=t_off,
                       dt=dt,
                       n_summary=n_summary,
                       _type=feature_type,
                       PCA_file=PCA_file)
        obs_stats = s.calc([{'data': obs}])
        n_summary = obs_stats.reshape(-1).shape[0]
        print("n summary: ", n_summary)
        m = []
        for i in range(n_processes):
            m.append(
                Ribon(stim,
                      dt,
                      dim_param,
                      pre_step,
                      SinglePathwayModel,
                      ID=i,
                      seed=np.random.randint(1000)))
        g = dg.MPGenerator(models=m,
                           prior=prior,
                           summary=s,
                           data_file_name=store_file)

    else:
        print("Model only supports [HH, Ribon], but got {}".format(model_name))

    # convenience
    prior_norm = True

    # Density
    # density = 'maf'
    assert density in ['maf', 'mog'], print(
        "Density should be in [mog, maf], but got {}".format(density))

    if pilot_file is not None:
        pilot_samples = load_data(pilot_file)

    # inference object
    if feature_type == 'Raw':
        n_filters = [64, 128, 256, 512]
    else:
        n_filters = ()
    res = infer.SNPEC(
        g,
        obs=obs_stats,
        n_hiddens=n_hiddens,
        seed=np.random.randint(1000),
        pilot_samples=pilot_samples,
        n_mades=n_mades,
        # prior_norm=prior_norm,
        density=density,
        # n_filters=n_filters,
    )

    # train
    log, _, posterior = res.run(
        n_train=n_train,
        n_rounds=n_rounds,
        minibatch=minibatch,
        epochs=epochs,
        silent_fail=False,
        proposal=proposal,
        val_frac=val_frac,
        verbose=True,
    )

    fig = plt.figure(figsize=(15, 5))
    val_loss_iter = log[0]['val_loss_iter']
    val_loss = log[0]['val_loss']
    loss = log[0]['loss']
    _len = loss.reshape(-1).shape[0]
    for i in range(1, len(log)):
        val_loss_iter = np.concatenate(
            (val_loss_iter, log[i]['val_loss_iter'] + _len))
        val_loss = np.concatenate((val_loss, log[i]['val_loss']))
        loss = np.concatenate((loss, log[i]['loss']))
        _len += log[i]['loss'].reshape(-1).shape[0]

    plt.plot(val_loss_iter, val_loss, lw=2, c='b', label='Val')
    plt.plot(loss, lw=2, c='r', label='Train')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(os.path.join(result_dir, 'loss.png'), dpi=400)

    prior_min = g.prior.lower
    prior_max = g.prior.upper
    prior_lims = np.concatenate(
        (prior_min.reshape(-1, 1), prior_max.reshape(-1, 1)), axis=1)

    posterior_samples = posterior[-1].gen(10000)
    np.save(os.path.join(result_dir, 'param_samples.npy'), posterior_samples)
    ###################
    # colors
    hex2rgb = lambda h: tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

    # RGB colors in [0, 255]
    col = {}
    col['GT'] = hex2rgb('30C05D')
    col['SNPE'] = hex2rgb('2E7FE8')
    col['SAMPLE1'] = hex2rgb('8D62BC')
    col['SAMPLE2'] = hex2rgb('AF99EF')

    # convert to RGB colors in [0, 1]
    for k, v in col.items():
        col[k] = tuple([i / 255 for i in v])
    ###################

    # posterior
    if true_params is not None:
        true_params = [true_params]
    else:
        true_params = []
    fig, axes = samples_nd(posterior_samples,
                           limits=prior_lims,
                           ticks=prior_lims,
                           labels=labels_params,
                           fig_size=(5, 5),
                           diag='kde',
                           upper='kde',
                           hist_diag={'bins': 50},
                           hist_offdiag={'bins': 50},
                           kde_diag={
                               'bins': 50,
                               'color': col['SNPE']
                           },
                           kde_offdiag={'bins': 50},
                           points=true_params,
                           points_offdiag={'markersize': 5},
                           points_colors=[col['GT']],
                           title='')

    plt.savefig(os.path.join(result_dir, 'posterior.png'), dpi=400)

    fig = plt.figure(figsize=(7, 5))

    # y_obs = obs
    t = np.arange(obs.shape[0])
    duration = np.max(t)

    num_samp = 200

    # sample from posterior
    x_samp = posterior[0].gen(n_samples=num_samp)

    # reject samples for which prior is zero
    ind = (x_samp >= prior_min) & (x_samp <= prior_max)
    params = x_samp[np.prod(ind, axis=1) == 1]

    num_samp = min(2, len(params[:, 0]))

    # simulate and plot samples
    V = np.zeros((len(t), num_samp))
    for i in range(num_samp):
        x = m[0].gen_single(params[i, :])
        V[:, i] = x['data']
        plt.plot(t,
                 V[:, i],
                 color=col['SAMPLE' + str(i + 1)],
                 lw=2,
                 label='sample ' + str(num_samp - i))

    # plot observation
    plt.plot(t, obs, '--', lw=2, label='observation')
    plt.xlabel('time (ms)')
    plt.ylabel('voltage (mV)')

    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1],
              labels[::-1],
              bbox_to_anchor=(1.3, 1),
              loc='upper right')

    ax.set_xticks([0, duration / 2, duration])
    # ax.set_yticks([-80, -20, 40])

    plt.savefig(os.path.join(result_dir, 'result.png'), dpi=400)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-model_name',
                        type=str,
                        help='The model to be simulated, within [HH, Ribon]')
    parser.add_argument('-n_summary', type=int, help='Number of summary stats')
    parser.add_argument('-n_processes',
                        type=int,
                        help='Number of processes for simulation')
    parser.add_argument(
        '-store_file',
        type=str,
        default=None,
        help='Where to store the simulation data, default: None')
    parser.add_argument(
        '-pilot_samples',
        type=int,
        default=None,
        help='Number of pilot samples, None for loading samples, default: None')
    parser.add_argument(
        '-pilot_file',
        type=str,
        default=None,
        help=
        'File path to the pilot samples, None for simulation online, default: None'
    )
    parser.add_argument(
        '-n_train',
        type=int,
        default=1000,
        help='Number of training samples per epoch, default: 1000')
    parser.add_argument('-n_rounds',
                        type=int,
                        default=2,
                        help='Number of training rounds, default: 2')
    parser.add_argument('-batch',
                        type=int,
                        default=128,
                        help='Batch size, default: 128')
    parser.add_argument('-epoch',
                        type=int,
                        default=128,
                        help='Training epoch per round, default: 1000')
    parser.add_argument('-val_frac',
                        type=float,
                        default=0.05,
                        help='Validation set fraction, default: 0.05')
    parser.add_argument('-dim_hiddens',
                        type=int,
                        default=100,
                        help='Hidden dimension, default: 100')
    parser.add_argument('-hidden_layer',
                        type=int,
                        default=2,
                        help='Hidden layers, default: 2')
    parser.add_argument('-n_mades',
                        type=int,
                        default=5,
                        help='Number of MADEs, default: 5')
    parser.add_argument(
        '-density',
        type=str,
        help='Density for estimation, should be within [mog, maf]')
    parser.add_argument(
        '-proposal',
        type=str,
        help=
        'Proposal to use, should be in [prior, gaussion, mog, atomic, atomic_comb]'
    )
    parser.add_argument('-feature',
                        type=str,
                        help='Feature to use, should be in [He, PCA, Raw]')
    parser.add_argument('-result_dir',
                        type=str,
                        help='Path to store the results')
    parser.add_argument('-PCA_file',
                        type=str,
                        default=None,
                        help='Path to PCA_file')
    parser.add_argument('-dt', type=int, help='dt')

    args = parser.parse_args()

    run(args)
