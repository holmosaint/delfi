#coding:UTF-8

# import h5py
import math
import dill
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from configures import *
from utils import get_peaks_index, get_func_from_parameters, normalize_trace
from filters import TemporalFilter, ContainerFilter
from models import Ribbon, RmaxRibbon
from photoreceptor import DA18 as Cone_MODEL
import optimize as O

# DATA =  h5py.File("../datas/FrankeEtAl_BCs_2017_v1.mat", "r")
RESP_CHECK = [
    (2000, 125),
    (5000, 315),
    (8000, 500),
    (10000, 640),
    (18100, 1130),
    (20000, 1280),
    (28000, 1740),
    (30000, 1860),
    (31998.999624859323, 2048),
]

DT = 0.1

[
    15.67777320402249, -0.056791206795922054, 1.1191557005173172,
    10.822462928458327, 128.08111364345706, 0.06406046475305106,
    0.6935847271389265, 0.058909332828580865, 8.98137607778845e-05,
    51.656473505260884, 19.826843632437477, 0.051296586456530686,
    132.0951533495297
]

[
    0.2867, 1.4255, 9.616, 0.067, 0.73, -0.6, 175., 9.74729, 0.061, 0.00012,
    0.0377, 50.12, 0.00188, 2., 5., 70.
]

[
    3.433325684022069, 4.566643659008289, 14.499384645924543,
    0.3753353767148771, 0.43221318387868524, -5.130981624566903,
    162.71439042342692, 20.368198422735595, 0.357587436146311,
    0.00043905120636321036, 0.07224261102611453, 308.7228832049189,
    0.009774450357724025, 30.045522639775164, 2.530232150130778,
    1071.3366307485592
]
[
    1.82042759508533, 4.652947158285871, 8.202554350214859, 0.39982186424442145,
    -0.20966383531977761, -3.249582221496651, 242.6780543380264,
    4.069556208192722, 0.5424813331526837, 0.0007236331063469786,
    0.057927785290579376, 359.48174076620444, 0.006837660918257103,
    -31.891726303407516, 6.045331127525978, 236.90869522818718
]
[
    3.244723254660908, 10.915303386957811, 13.878967141197133,
    0.06062008599349561, -0.4843435671050944, -3.776928715874104,
    1630.118166390761, 13.132666823390942, 0.44096305505105343,
    0.0009099725948107005, 0.007782585551281132, 23.414101693288195,
    0.009519029863395442, -29.208232385298352, 11.509424020391268,
    3306.121801011765
]


class SinglePathwayModel:
    """NTGCTNTGC 
    Sigmoid - TemporalFilter - Ribbon - TemporalFilter - Sigmoid - TemporalFilter - Ribbon
    function for the time constant of gain is sigmoid

    parameters:
    -----------------------
    #0: the half value of 1st sigmoid function
    #1: the slope value of 1st sigmoid function

    #2: the time constant of 1st temporal filter

    #3: the k value of the reverse hill function which defines the stable gain in the 1st ribbon
    #4: the maimum value of the reverse hill function which defines the stable gain in the 1st ribbon
    #5: the half value of sigmoid function which defines the time constant during gain control in the 1st ribbon
    #6: the slope value of sigmoid function which defines the time constant during gain control in the 1st ribbon
    #7: the maximum value of id function which defines the time constant during gain control in the 1st ribbon

    #8: the time constant of 2nd temporal filter

    #9: the half value of 2nd sigmoid function
    #10: the slope value of 2nd sigmoid function
    #11: the maximum value of 2nd sigmoid function

    #12: the time constant of 3rd temporal filter

    #13: the k value of the reverse hill function which defines the stable gain in the 2nd ribbon
    #14: the maimum value of the reverse hill function which defines the stable gain in the 2nd ribbon
    #15: the half value of sigmoid function which defines the time constant during gain control in the 2nd ribbon
    #16: the slope value of sigmoid function which defines the time constant during gain control in the 2nd ribbon
    #17: the maximum value of id function which defines the time constant during gain control in the 2nd ribbon

    #18: the time constant of 4th temporal filter

    if ribbon is RMAX:
    #5~6 #15~16 is discard
    #4 & #14 is fixed to 1.

    if feedback_mode != 0:
    #19 the time constant for feedback in the 1st ribbon synapse
    #20 the weight for feedback in the 1st ribbon synapse
    #21 the time constant for feedback in the 2nd ribbon synapse
    #22 the weight for feedback in the 2nd ribbon synapse

    
    feedback_mode:
    ---------------------
    #0: none
    #1: inhibitate [Ca]
    #2: inhibitate release
    """

    def __init__(self,
                 parameters,
                 PHOTO=False,
                 feedback_mode=0,
                 RMAX=False,
                 single=False,
                 dt=DT):
        self.parameters = parameters
        self.single = single
        if PHOTO:
            self.cone = Cone_MODEL(*parameters[:5], dt=dt)
            parameters = parameters[5:]
        if single:
            self.sigmoid1 = get_func_from_parameters("sigmoid", parameters[:3])
            self.temporal1 = ContainerFilter(lambda x: parameters[3], dt=dt)
            if RMAX:
                gain_infi1 = get_func_from_parameters("rhill",
                                                      [parameters[4]] + [1.])
                self.ribbon1 = RmaxRibbon(1. / parameters[5],
                                          gain_infi,
                                          RRP=1. / dt,
                                          dt=dt)
            else:
                gain_infi1 = get_func_from_parameters("rhill", parameters[4:6])
                gain_tau1 = get_func_from_parameters("sigmoid", parameters[6:9])
                self.ribbon1 = Ribbon(1.,
                                      gain_infi1,
                                      gain_tau1,
                                      RRP=1. / dt,
                                      dt=dt)
        else:
            self.sigmoid1 = get_func_from_parameters(
                "sigmoid",
                list(parameters[:2]) + [1.])
            self.temporal1 = ContainerFilter(lambda x: parameters[2], dt=dt)
            if RMAX:
                gain_infi1 = get_func_from_parameters("rhill",
                                                      [parameters[3]] + [1.])
                self.ribbon1 = RmaxRibbon(1. / parameters[4],
                                          gain_infi1,
                                          RRP=1. / dt,
                                          dt=dt)
                self.temporal2 = ContainerFilter(lambda x: parameters[5], dt=dt)
                self.sigmoid2 = get_func_from_parameters(
                    "sigmoid", parameters[6:9])
                self.temporal3 = ContainerFilter(lambda x: parameters[9], dt=dt)
                gain_infi2 = get_func_from_parameters("rhill",
                                                      [parameters[10]] + [1.])
                self.ribbon2 = Ribbon(1. / parameters[11],
                                      gain_infi2,
                                      RRP=1. / dt,
                                      dt=dt)
                self.temporal4 = ContainerFilter(lambda x: parameters[12],
                                                 dt=dt)
            else:
                gain_infi1 = get_func_from_parameters("rhill", parameters[3:5])
                gain_tau1 = get_func_from_parameters("sigmoid", parameters[5:8])
                self.ribbon1 = Ribbon(1.,
                                      gain_infi1,
                                      gain_tau1,
                                      RRP=1. / dt,
                                      dt=dt)
                self.temporal2 = ContainerFilter(lambda x: parameters[8], dt=dt)
                self.sigmoid2 = get_func_from_parameters(
                    "sigmoid", parameters[9:12])
                self.temporal3 = ContainerFilter(lambda x: parameters[12],
                                                 dt=dt)
                gain_infi2 = get_func_from_parameters("rhill",
                                                      parameters[13:15])
                gain_tau2 = get_func_from_parameters("sigmoid",
                                                     parameters[15:18])
                self.ribbon2 = Ribbon(1.,
                                      gain_infi2,
                                      gain_tau2,
                                      RRP=1. / dt,
                                      dt=dt)
                self.temporal4 = ContainerFilter(lambda x: parameters[18],
                                                 dt=dt)
        self.dt = dt

        self.fb_mode = feedback_mode
        if feedback_mode != 0:
            self.fb_temporal1 = ContainerFilter(lambda x: parameters[19], dt=dt)
            self.fb_temporal2 = ContainerFilter(lambda x: parameters[21], dt=dt)
            self.fb_weight1 = parameters[20]
            self.fb_weight2 = parameters[22]

    def run(self, trace):
        rs = []
        if hasattr(self, "cone"):
            trace = self.cone.run(trace)[0]
            # normalize
            trace = (trace - trace.min()) / (trace.max() - trace.min())
            rs.append(trace)
        sigmoid1 = self.sigmoid1(trace)
        temporal1 = self.temporal1.filter(sigmoid1)
        rs = [temporal1, sigmoid1] + rs
        if self.fb_mode == 0:
            if self.single:
                ribbon1 = self.ribbon1.run(temporal1)
                return ribbon1 + rs

            ribbon1 = self.ribbon1.run(temporal1)
            temporal2 = self.temporal2.filter(ribbon1[0])
            sigmoid2 = self.sigmoid2(temporal2)
            temporal3 = self.temporal3.filter(sigmoid2)
            ribbon2 = self.ribbon2.run(temporal3)
            return ribbon2 + [temporal3, sigmoid2, temporal2] + ribbon1 + rs
        if self.fb_mode == 1:
            # feedback inhibitation on [Ca]
            # (temporal filter for input of ribbon synapse)
            pass
        elif self.fb_mode == 2:
            # feedback inhibitation on release
            pass
        else:
            raise Exception("UNKNOWN feedback mode for SinglePathwayModel")


def get_data_idx(value_name, cluster_id=1):
    d = np.array(DATA[value_name])
    if d.shape[0] == 1:
        return d[0]
    index = np.where(DATA['cluster_idx'][0] == cluster_id)[0]
    value = np.zeros(d.shape[1:])
    for i in index:
        value += d[i]
    return value / index.shape[0]


def get_data_experiment(cluster_id,
                        data_mat,
                        idx_mat,
                        data_label,
                        data_id,
                        data_key,
                        dt=DT):
    time = np.array(idx_mat['chirp_time']).flatten()
    cato_indices = np.where(np.array(idx_mat['cluster_idx']) == cluster_id)[0]
    cato_indices = set(cato_indices)
    data_indices = np.where(np.array(data_mat[data_label]) == data_id)[0]
    data_indices = set(data_indices)
    indices = np.array(list(cato_indices & data_indices), dtype=int)
    nonan_indices = []
    for index in indices:
        if True not in np.isnan(data_mat[data_key][:, index]):
            nonan_indices.append(index)
    print("%d valid trace(s)" % len(nonan_indices))
    if len(nonan_indices) == 0:
        return None
    data = np.array(data_mat[data_key]).T[nonan_indices].mean(axis=0)
    time = time * 1000.
    # return time, data, nonan_indices, indices
    time = np.array([0.] + time.tolist())
    data = normalize_trace(data, data[0])
    data = np.array([0.] + data.tolist())
    x = np.arange(0, 31992, dt)
    func = interp1d(time, data, kind='quadratic')
    y = np.vectorize(func)(x)
    return y


def get_linear_filter(cluster_id=1, dt=DT, interp_kind='quadratic'):
    weights = get_data_idx('rf_tc', cluster_id)
    time = DATA['rf_time'][:, 0] * 1000  # (s) -> (ms)
    func = interp1d(time, weights, kind=interp_kind)
    x = np.arange(int(time[0]), int(time[-1]), dt)[::-1]
    print(np.where(abs(x - 0) < dt / 10.))
    y = np.vectorize(func)(x)
    shifted = len(x) - (np.where(abs(x - 0) < dt / 100.)[0][0])
    return TemporalFilter(y, shifted)


def get_stimulus(DATATYPE=0):
    if DATATYPE == 0:
        stim = np.loadtxt('../datas/Franke/chirp_stim.txt')
    elif DATATYPE == 1:
        stim = np.loadtxt('../datas/Franke/DA18_normalized_0.1.txt')
    return stim


def get_response(cluster_id, label='lchirp_avg', interp_kind="quadratic"):
    if type(cluster_id) == type(5):
        resp = get_data_idx(label, cluster_id)
    else:
        resp = np.array(cluster_id)

    time = np.array(DATA['chirp_time'][0]) * 1000
    time = np.array([0] + time.tolist())
    resp = np.array([0] + resp.tolist())
    x = np.arange(0, int(time[-1]), step=0.1)
    func = interp1d(time, resp, kind=interp_kind)
    y = func(x)
    return y


def save_response(prefix="../datas/Franke/lchirp_control_resp",
                  label='lchirp_avg'):
    for i in range(1, 15):
        resp = get_response(i, label)
        np.savetxt("%s_%d.txt" % (prefix, i), resp)


def get_data_pair_lchirp(cluster_id, buffer_time=10000, DATATYPE=0, dt=DT):
    stim = get_stimulus(DATATYPE)
    response = np.loadtxt("../datas/Franke/lchirp_control_resp_%d.txt" %
                          cluster_id)
    stim = stim[::int(dt / DT)]
    response = response[::int(dt / DT)]
    response -= response[:int(2000 / dt)].mean()
    n = min(len(stim), len(response))
    stim = stim[:n]
    response = response[:n]
    stim = [stim[0]] * int(buffer_time / dt) + stim.tolist()
    pre_step = int(buffer_time / dt)
    return pre_step, stim, response


def is_cell_off(cluster_id):
    return cluster_id < 6


def fit_CC(cluster_id,
           x0=None,
           ext_model_parameters={'fixed_parameters': []},
           DATATYPE=0,
           dt=DT):
    pre_step, stim, resp = get_data_pair_lchirp(cluster_id,
                                                DATATYPE=DATATYPE,
                                                dt=dt)
    import nr
    MODEL = nr.NTGCNTGC
    if x0 is None:
        p_base = [
            0.01, -1, 50, 50, 0.01, 0.01, 0.01, 0, 0.1, 50, 50, 0.01, 0.01
        ]
        p_range = [1, 2, 500, 500, 2, 2, 1, 1, 2, 500, 500, 2, 2]
        x0 = np.random.random(
            len(p_base)) * np.array(p_range) + np.array(p_base)
    return fit(MODEL, x0, False, cluster_id, (pre_step, stim, resp),
               ext_model_parameters, dt)


def fit(MODEL,
        x0,
        need_init,
        cluster_id,
        data,
        ext_model_parameters={},
        fname=None,
        dt=DT):
    pre_step, stim, resp = tuple(data)
    pre_time = pre_step * dt
    return O.optimize_traces(MODEL,
                             x0, [stim], [resp],
                             pre_time,
                             1,
                             dt,
                             ext_model_parameters,
                             need_init,
                             True,
                             evaluate_method=O.evaluate_abs_normalized,
                             record_fname=fname)


def generate_pca(n_feature=6):
    data = np.array(DATA['lchirp_avg'])
    from sklearn.decomposition import SparsePCA
    pca = SparsePCA(n_feature, normalize_components=True)
    pca.fit(data)
    return pca
    with open("Franke_lchirp_pca.pkl", "w+") as f:
        dill.dump(pca, f)


def get_response_pca_features(response, pca):
    features = pca.transform(np.array(response).reshape((1, -1)))
    return features[0]


# running cases for Single SinglePathwayModel


def sample_parameters(fname, TYPE="ON", max_n=None, n=5):
    if TYPE == "ON":
        x0 = np.linspace(0.5, 4, n)  # half of sigmoid
        x1 = np.linspace(0.1, 3, n)  # slope of sigmoid
        x2 = np.linspace(0.1, 5, n)  # maximum of sigmoid
        x3 = np.linspace(0.1, 50, n)  # tau fo temporal filter
        x4 = np.linspace(0.01, 2, n)  # k of gain infi
        x5 = np.linspace(0.2, 3, n)  # m of gain infi
        x6 = np.linspace(-2, +2, n)  # half of gain tau
        x7 = np.linspace(-1, +1, n - n % 2)  # slope of gain tau
        x8 = np.linspace(100, 1500, n)  # max of gain tau
        xs = [x0, x1, x2, x3, x4, x5, x6, x7, x8]
    all_n = 1
    for x in xs:
        all_n *= len(x)
    import itertools, dill
    all_pss = list(itertools.product(*xs))
    if max_n is not None and max_n < all_n:
        mask = np.random.choice(range(all_n), replace=False, size=max_n)
        selected = np.array(all_pss)[mask]
    else:
        selected = np.array(all_pss)
    with open(fname, "w") as f:
        dill.dump(selected, f)


def run_parameters(fname, pss_fname, ext, MODEL, n_pool, pre_time=20000):
    import dill
    stim = get_stimulus(DATATYPE=1)
    stim = stim[::20]  #(2ms)
    pre = int(pre_time / 2)
    stim = np.array([stim[0]] * pre + stim.tolist())
    if type(pss_fname) is str:
        pss = dill.load(open(pss_fname))
    else:
        pss = pss_fname
    from pathos.multiprocessing import ProcessingPool as Pool
    pool = Pool(n_pool)

    def run(ps):
        with open(fname, "a") as f:
            model = MODEL(ps, dt=2, **ext)
            r = model.run(stim)[0][pre:]
            line = " ".join(np.array(list(ps) + list(r), dtype=str)) + "\n"
            f.write(line)

    pool.map(run, pss)


if __name__ == "__main__":
    # if DATATYPE == 1, we actually sample OFF type ....
    run_parameters("/media/retina/Seagate Backup Plus Drive/SAMPLE_ON_DATA.txt",
                   "res/Franke/SAMPLE_ON.pkl", {"single": True},
                   SinglePathwayModel, 50)
"""
#TODO
Dual-Pathway Model
why (B) conflicts with the common sense of OFF-BC
different taus in (B) absolutely 
<- might also two pathways for release or two model for release
and several tau is very larg (A/C) how to affect after-response?
or FEEDBACK??
or
ONE Pathway is very slow and becomes majoir in the dark (~ROD)
ANOTHER one appears when light is high (~CONE)
or otherwise <- DYNAMIC system <- like rod-cone interaction pathway
or 

center - surround
center: cone - NTGC ---(T)------- T --|
       (rod/cone - NTGC -(T)-|) |     |
                                      |---NTGC -> Release
surround: cone - NTGC --(T)------ T --|
       (rod/cone - NTGC -(T)-|)

"""
