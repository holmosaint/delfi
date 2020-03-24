#!python
#coding:UTF-8
"""
Models for Gain Control
State (Ribbon) + Transient (Feedback)

"""

import numpy as np

import dill

from .configures import *
from .utils import get_func_from_parameters, get_linear_func, get_interp_func
from .filters import SlowFilter, ContainerFilter


class GainControler:
    """
    Framework for Gain-Contorler
    """

    def __init__(self,
                 full_release_func,
                 gain_infi_func,
                 gain_tau_func,
                 output_func=lambda p, g: p * g,
                 RRP=1.,
                 dt=DT):
        # function for full release: get full release from input
        # e.g. full_release_func=lambda input, RRP: 0.3*input*RRP  (n/ms)
        self.full_release_func = np.vectorize(full_release_func)

        # function for gain infinity: get stable gain when stimulus last for a long time
        # e.g. gain_infi_func=lambda stim: 1./(1+stim)  (1.)
        self.gain_infi_func = np.vectorize(gain_infi_func)

        # function for gain time constant: get tau for system reach the stable state
        # e.g. gain_tau_func=lambda stim: 100.  (ms)
        self.gain_tau_func = np.vectorize(gain_tau_func)

        # function for output: decide release rate from full_release and gain
        # default is use output = full*gain  (n/ms)
        self.output_func = np.vectorize(output_func)

        self.g_func = lambda infi, tau, g0: infi + (g0 - infi) * np.exp(-self.dt
                                                                        / tau)

        def resupply_func(actual_release, tau, infi, gain):
            return actual_release + RRP * (infi - gain) / tau

        self.resupply_func = np.vectorize(resupply_func)

        self.RRP = float(RRP)
        self.dt = float(dt)
        self.g = 0.5

    def get_expect_resupply_func(self):

        def func(ca, gain):
            A = self.full_release_func(
                ca, self.RRP) - self.RRP / self.gain_tau_func(ca)
            B = self.RRP * self.gain_infi_func(ca) / self.gain_tau_func(ca)
            return (A * gain + B)

        return np.vectorize(func)

    def init(self, init_g=0.5):
        self.g = init_g

    def run(self, trace, return_rate=False):
        full_releases = self.full_release_func(trace, self.RRP)
        gain_infis = self.gain_infi_func(trace)
        gain_taus = self.gain_tau_func(trace)
        gain_taus[gain_taus < self.dt] = self.dt
        gains = []
        g = self.g
        for infi, tau in zip(gain_infis, gain_taus):
            g = self.g_func(infi, tau, g)
            gains.append(g)
        gains = np.array(gains)
        release_rate = self.output_func(full_releases, gains)
        resupply_rate = self.resupply_func(release_rate, gain_taus, gain_infis,
                                           gains)
        if return_rate:
            return [
                release_rate, gains, resupply_rate, full_releases, gain_infis,
                gain_taus
            ]
        release = release_rate * self.dt
        resupply = resupply_rate * self.dt
        self.g = g
        return [release, gains, resupply, full_releases, gain_infis, gain_taus]

    def run_random(self, trace):
        gain_infis = self.gain_infi_func(trace)
        gain_taus = self.gain_tau_func(trace)
        gains = []
        releases = []
        resupplies = []
        fulls = []
        nrrp = int(self.RRP / 2)
        gain = nrrp / self.RRP
        for infi, tau, stim in zip(gain_infis, gain_taus, trace):
            full = self.full_release_func(stim, self.RRP)
            fulls.append(full)
            release_rate = self.output_func(full, gain)
            resupply_rate = self.resupply_func(release_rate, tau, infi, gain)
            release = release_rate * self.dt
            resupply = resupply_rate * self.dt
            if nrrp > 0:
                p = release / nrrp
                release = np.where(np.random.rand(nrrp) < p)[0].shape[0]
            else:
                release = 0
            if self.RRP > nrrp:
                p = resupply / (self.RRP - nrrp)
                resupply = np.where(np.random.rand(int(self.RRP - nrrp)) < p)
                resupply = resupply[0].shape[0]
            else:
                resupply = 0
            delta = -release + resupply
            if nrrp + delta < 0:
                release = nrrp + resupply
                nrrp = 0
            elif nrrp + delta > self.RRP:
                resupply = self.RRP - nrrp - release
                nrrp = self.RRP
            else:
                nrrp += delta
            gain = nrrp / self.RRP
            gains.append(gain)
            releases.append(release)
            resupplies.append(resupply)
        gains = np.array(gains)
        releases = np.array(releases)
        resupplies = np.array(resupplies)
        fulls = np.array(fulls)
        return [releases, resupplies, gains, fulls, gain_infis, gain_taus]

    def run_traces(self, traces, repeat=1, random=False):
        res = []
        for trace in traces:
            rs = []
            for i in range(repeat):
                if random:
                    r = self.run_random(trace)
                else:
                    r = self.run(trace)
                rs.append(r)
            res.append(rs)
        return res


class DoubleGainControler:
    """Two mechanisms for gain controler:
    1. instant gain decide by the change of input
    2. infi gain decide by current input"""

    def __init__(self, change2gain_func, tau, infi_gain_func, dt=DT):
        self.change2gain_func = change2gain_func
        self.infi_gain_func = np.vectorize(infi_gain_func)
        self.dt = dt
        self.gain_buffer = ContainerFilter(lambda x: tau, dt=dt)

    def run(self, trace, init_gain=0.5):
        trace = np.array(trace)
        change_gains = self.change2gain_func(trace)
        infi_gains = self.infi_gain_func(trace)[1:]
        gains = [init_gain]
        for change_gain, infi_gain in zip(change_gains, infi_gains):
            if change_gain == 0:
                new_gain = self.gain_buffer.filter([infi_gain])[0]
            else:
                new_gain = change_gain
                self.gain_buffer.init(new_gain)
            gains.append(new_gain)
        gains = np.array(gains)
        return [trace * gains, gains, change_gains, infi_gains]


class TransientGainControler(DoubleGainControler):

    def __init__(self, feedback_weight, tau, dt=DT):
        self.input_buffer = ContainerFilter(lambda x: tau, dt=dt)
        self.weight = feedback_weight
        self.infi_gain = 1 / (1 - self.weight)
        self.func = np.vectorize(lambda s, b: 0 if abs(s - b) < 1e-9 else 1 +
                                 self.infi_gain * self.weight / (s / b))

        def change2gain_func(trace):
            base = self.input_buffer.filter(trace)
            return np.array(
                [self.func(i, j) for i, j in zip(trace[1:], base[:-1])])

        infi_gain_func = lambda x: 1 / (1 - self.weight)
        DoubleGainControler.__init__(self,
                                     change2gain_func,
                                     tau,
                                     infi_gain_func,
                                     dt=dt)

    def run(self, trace):
        return DoubleGainControler.run(self, trace, self.infi_gain)

    def init(self, stim):
        self.input_buffer.init(stim)
        self.gain_buffer.init(self.infi_gain)
        return stim * self.infi_gain


class Ribbon(GainControler):
    """
    Ribbon Synapse Model
    """

    def __init__(self,
                 release_k=0.03,
                 gain_infi_func=lambda x: 0.4 / (x + 0.4),
                 gain_tau_func=lambda x: 50 / (x + 0.0001),
                 RRP=100,
                 dt=DT):
        full_release = lambda stim, RRP: release_k * stim * RRP
        GainControler.__init__(self,
                               full_release,
                               gain_infi_func,
                               gain_tau_func,
                               RRP=RRP,
                               dt=dt)


class FlexibleRibbon(Ribbon):

    def __init__(self,
                 infi_values,
                 tau_values,
                 max_input,
                 release_k=1,
                 RRP=1,
                 dt=DT):
        for i in infi_values:
            if i <= 0:
                raise Exception("Gain must >= 0")
        infi_func = get_interp_func(infi_values, 0, max_input)
        tau_func = get_interp_func(tau_values, 0, max_input)
        Ribbon.__init__(self, release_k, infi_func, tau_func, RRP, dt=dt)


class RmaxRibbon(Ribbon):
    """ A Rmax-fix Ribbon model """

    def __init__(self,
                 release_k=0.03,
                 gain_infi_func=lambda x: 0.4 / (x + 0.4),
                 RRP=100,
                 dt=DT):
        full_release = lambda stim, RRP: release_k * stim * RRP
        gain_tau_func = lambda x: (1 - gain_infi_func(x)) / (release_k * x)
        GainControler.__init__(self,
                               full_release,
                               gain_infi_func,
                               gain_tau_func,
                               RRP=RRP,
                               dt=dt)


#######
#
# Models to fit adaptive model (LNK/GLM)
# Based on Ribbon
#
#######


class GC:

    def __init__(self, parameters, gain_infi_func, gain_tau_type, dt=DT):
        gain_tau_func = get_func_from_parameters(gain_tau_type, parameters)
        self.ribbon = Ribbon(1, gain_infi_func, gain_tau_func, RRP=1., dt=dt)

    def run(self, init_stim, trace):
        init_g = self.ribbon.gain_infi_func(init_stim)
        res = self.ribbon.run(trace, init_g, return_rate=True)
        return res


class TGC(GC):
    """
    SlowTemporalFilter + Gain Controler 
    """

    def __init__(self, parameters, gain_infi_func, gain_tau_type, dt=DT):
        # print(parameters)
        tau = parameters[0]
        self.temporal = ContainerFilter(lambda x: tau, dt=dt)
        GC.__init__(self, parameters[1:], gain_infi_func, gain_tau_type, dt=dt)

    def run(self, init_stim, trace):
        if isinstance(self.temporal, ContainerFilter):
            self.temporal.init(init_stim)
        else:
            self.temporal.buffer[:] = init_stim
        temp = self.temporal.filter(trace)
        res = GC.run(self, init_stim, temp)
        return res + [temp]


class TGCT(TGC):

    def __init__(self, parameters, gain_infi_func, gain_tau_type, dt=DT):
        tau2 = parameters[-1]
        self.temporal2 = ContainerFilter(lambda x: tau2, dt=dt)
        TGC.__init__(self, parameters[:-1], gain_infi_func, gain_tau_type, dt)

    def run(self, init_stim, trace):
        res = TGC.run(self, init_stim, trace)
        self.temporal2.init(res[0][0])
        r = self.temporal2.filter(res[0])
        return [r] + res


class TGCT_Free(TGCT):

    def __init__(self,
                 parameters,
                 n_params4infi,
                 gain_infi_type,
                 gain_tau_type,
                 dt=DT):
        infi_params = parameters[:n_params4infi]
        gain_infi_func = get_func_from_parameters(gain_infi_type, infi_params)
        TGCT.__init__(self, parameters[n_params4infi:], gain_infi_func,
                      gain_tau_type, dt)


class TGCGC:

    def __init__(self, parameters, gain_infi_func, gain_tau_type, dt=DT):
        k = parameters[0]
        infi1 = lambda x: k / (x + k)

        def infi2(x):
            o = k * x / (k - x)  # original input  (o*k/(o+k) = x)
            expected_output = o * gain_infi_func(o)  # supposed output
            gain = expected_output / x
            # print(x, o, expected_output, gain)
            return gain

        n = int((len(parameters) - 2) / 2)
        ps1 = parameters[1:2 + n]
        ps2 = parameters[2 + n:]
        self.tgc1 = TGC(ps1, infi1, gain_tau_type, dt=dt)
        self.gc2 = GC(ps2, infi2, gain_tau_type, dt=dt)

    def run(self, init_stim, trace):
        tgc1_res = self.tgc1.run(init_stim, trace)
        init_tgc1 = self.tgc1.ribbon.gain_infi_func(init_stim) * init_stim
        gc2_res = self.gc2.run(init_tgc1, tgc1_res[0])
        return gc2_res + tgc1_res


class TGCTGC:

    def __init__(self, parameters, gain_infi_func, gain_tau_type, dt=DT):
        k = parameters[0]
        infi1 = lambda x: k / (x + k)

        def infi2(x):
            o = k * x / (k - x)  # original input  (o*k/(o+k) = x)
            expected_output = o * gain_infi_func(o)  # supposed output
            gain = expected_output / x
            # print(x, o, expected_output, gain)
            return gain

        n = int((len(parameters) - 1) / 2)
        ps1 = parameters[1:1 + n]
        ps2 = parameters[1 + n:]
        self.tgc1 = TGC(ps1, infi1, gain_tau_type, dt=dt)
        self.tgc2 = TGC(ps2, infi2, gain_tau_type, dt=dt)

    def run(self, init_stim, trace):
        tgc1_res = self.tgc1.run(init_stim, trace)
        init_tgc1 = self.tgc1.ribbon.gain_infi_func(init_stim) * init_stim
        tgc2_res = self.tgc2.run(init_tgc1, tgc1_res[0])
        return tgc2_res + tgc1_res


class TGCTGCT(TGCTGC):

    def __init__(self, parameters, gain_infi_func, gain_tau_type, dt=DT):
        tau = parameters[0]
        self.temporal = ContainerFilter(lambda x: tau, dt=dt)
        TGCTGC.__init__(self,
                        parameters[1:],
                        gain_infi_func,
                        gain_tau_type,
                        dt=dt)

    def run(self, init_stim, trace):
        r = TGCTGC.run(self, init_stim, trace)
        t = self.temporal.filter(r[0])
        return [t] + r


class TGCTGCTGC:

    def __init__(self,
                 parameters,
                 gain_infi_func,
                 gain_tau_type,
                 dt=DT,
                 max_x=100):
        k1 = parameters[0]
        k2 = parameters[1]
        infi1 = np.vectorize(lambda x: k1 / (x + k1))
        infi2 = np.vectorize(lambda x: k2 / (x + k2))
        xs = np.arange(0, max_x, 0.001)
        ys = xs * infi1(xs)
        ys = ys * infi2(ys)
        from scipy.interpolate import interp1d
        output_reverse_func = interp1d(ys, xs, kind="cubic")
        self.output_reverse_func = output_reverse_func

        def infi3(pre_output):
            if pre_output > ys[-1]:
                pre_output = ys[-1]
            origin_input = output_reverse_func(pre_output)
            expected_output = origin_input * gain_infi_func(origin_input)
            gain = expected_output / pre_output
            return gain

        self.infi1 = infi1
        self.infi2 = infi2
        n = int((len(parameters) - 2) / 3)
        ps1 = parameters[2:2 + n]
        ps2 = parameters[2 + n:2 + n * 2]
        ps3 = parameters[2 + n * 2:]
        self.tgc1 = TGC(ps1, infi1, gain_tau_type, dt=dt)
        self.tgc2 = TGC(ps2, infi2, gain_tau_type, dt=dt)
        self.tgc3 = TGC(ps3, infi3, gain_tau_type, dt=dt)

    def run(self, init_stim, trace):
        r1 = self.tgc1.run(init_stim, trace)
        init_r2 = init_stim * self.infi1(init_stim)
        r2 = self.tgc2.run(init_r2, r1[0])
        init_r3 = init_r2 * self.infi2(init_r2)
        # print(init_stim, trace[-1], init_r3, max(r2[0]), min(r2[0]))
        r3 = self.tgc3.run(init_r3, r2[0])
        return r3 + r2 + r1


class SumTGC:
    """ output is the sum of two TGCs """

    def __init__(self, parameters, gain_infi_func, gain_tau_type, dt=DT):
        w1 = parameters[0]
        if w1 < 0.:
            w1 = 0.
        elif w1 > 1.:
            w1 = 1.
        k = parameters[1]
        infi1 = lambda x: k / (x + k)

        def infi2(x):
            o1 = x * infi1(x) * w1
            expected_output = x * gain_infi_func(x)
            gain = (expected_output - o1) / (x * (1 - w1))
            return gain

        self.w1 = w1
        n = int((len(parameters) - 2) / 2)
        ps1 = parameters[2:2 + n]
        ps2 = parameters[2 + n:]
        self.tgc1 = TGC(ps1, infi1, gain_tau_type, dt=dt)
        self.tgc2 = TGC(ps2, infi2, gain_tau_type, dt=dt)

    def run(self, init_stim, trace):
        tgc1_res = self.tgc1.run(init_stim, trace)
        tgc2_res = self.tgc2.run(init_stim, trace)
        r = tgc1_res[0] * self.w1 + tgc2_res[0] * (1 - self.w1)
        return [r] + tgc1_res + tgc2_res


class SumTGC_FB(SumTGC):
    """ SumTGC + a TransientGain feedback + a temproal"""

    def __init__(self, parameters, gain_infi_func, gain_tau_type, dt=DT):
        self.output = ContainerFilter(lambda x: parameters[0], dt=dt)
        fb_weight, fb_tau = tuple(parameters[1:3])
        self.fb = TransientGainControler(fb_weight, fb_tau, dt=dt)
        self.gain_infi4sum = lambda x: gain_infi_func(x) / self.fb.infi_gain
        SumTGC.__init__(self,
                        parameters[3:],
                        self.gain_infi4sum,
                        gain_tau_type,
                        dt=DT)

    def run(self, init_stim, trace):
        r_sum = SumTGC.run(self, init_stim, trace)
        init_sum_output = init_stim * self.gain_infi4sum(init_stim)
        self.output.init(init_sum_output * self.fb.infi_gain)
        self.fb.init(init_sum_output)
        r_fb = self.fb.run(r_sum[0])
        r_output = self.output.filter(r_fb[0])
        return [r_output] + r_fb + r_sum


class Sum3TGC:

    def __init__(self, parameters, gain_infi_func, gain_tau_type, dt=DT):
        w1 = parameters[0]
        w2 = parameters[1]
        k1 = parameters[2]
        k2 = parameters[3]
        if w1 < 1e-4:
            w1 = 1e-4
        elif w1 > 0.9999:
            w1 = 0.9999
        if w2 < 1e-4:
            w2 = 1e-4
        elif w2 > 0.9999 - w1:
            w2 = 0.9999 - w1
        self.w1 = w1
        self.w2 = w2
        infi1 = lambda x: k1 / (x + k1)
        infi2 = lambda x: k2 / (x + k2)

        def infi3(x):
            o1 = x * infi1(x) * w1
            o2 = x * infi2(x) * w2
            expected_output = x * gain_infi_func(x)
            gain = (expected_output - o1 - o2) / (x * (1 - w1 - w2))
            return gain

        n = int((len(parameters) - 4) / 3)
        ps1 = parameters[4:4 + n]
        ps2 = parameters[4 + n:4 + n * 2]
        ps3 = parameters[4 + n * 2:]
        self.tgc1 = TGC(ps1, infi1, gain_tau_type, dt=dt)
        self.tgc2 = TGC(ps2, infi2, gain_tau_type, dt=dt)
        self.tgc3 = TGC(ps3, infi3, gain_tau_type, dt=dt)

    def run(self, init_stim, trace):
        tgc1_r = self.tgc1.run(init_stim, trace)
        tgc2_r = self.tgc2.run(init_stim, trace)
        tgc3_r = self.tgc3.run(init_stim, trace)
        r = tgc1_r[0] * self.w1 + tgc2_r[0] * self.w2 + tgc3_r[0] * (
            1 - self.w1 - self.w2)
        return [r] + tgc1_r + tgc2_r + tgc3_r


class TGT:
    """ TransientGainControler + temporal filter """

    def __init__(self, parameters, dt=DT):
        self.gc = TransientGainControler(*parameters[:2], dt=dt)
        self.temporal = ContainerFilter(lambda x: parameters[-1], dt=dt)

    def run(self, init_stim, trace):
        init_run = self.gc.init(init_stim)
        self.temporal.init(init_run)
        r = self.gc.run(trace)
        f = self.temporal.filter(r[0])
        return [f] + r


class SumTGCTGT:
    """ output is the sum of TGC and TGT """

    def __init__(self,
                 parameters,
                 gain_infi_func,
                 gain_tau_type,
                 w1=None,
                 dt=DT):
        if w1 is None:
            w1 = parameters[0]
            parameters = parameters[1:]
        if w1 < 0.0001:
            w1 = 0.0001
        elif w1 > 0.9999:
            w1 = 0.9999
        self.w1 = w1
        self.tgt = TGT(parameters[-3:], dt=dt)

        def infi(x):
            gain_ta = self.tgt.gc.infi_gain_func(x)
            o2 = x * gain_ta * (1 - w1)
            expected_output = x * gain_infi_func(x)
            gain = (expected_output - o2) / (x * w1)
            return gain

        ps = parameters[:-3]
        self.tgc = TGC(ps, infi, gain_tau_type, dt=dt)

    def run(self, init_stim, trace):
        r1 = self.tgc.run(init_stim, trace)
        r2 = self.tgt.run(init_stim, trace)
        r = r1[0] * self.w1 + r2[0] * (1 - self.w1)
        return [r] + r1 + r2


class SumTGCTGT_FB(SumTGCTGT):
    """ SumTGCTGT + a TransientGain feedback + a temproal"""

    def __init__(self, parameters, gain_infi_func, gain_tau_type, dt=DT):
        self.output = ContainerFilter(lambda x: parameters[0], dt=dt)
        fb_weight, fb_tau = tuple(parameters[1:3])
        self.fb = TransientGainControler(fb_weight, fb_tau, dt=dt)
        self.gain_infi4sum = lambda x: gain_infi_func(x) / self.fb.infi_gain
        SumTGCTGT.__init__(self,
                           parameters[3:],
                           self.gain_infi4sum,
                           gain_tau_type,
                           dt=DT)

    def run(self, init_stim, trace):
        r_sum = SumTGCTGT.run(self, init_stim, trace)
        init_sum_output = init_stim * self.gain_infi4sum(init_stim)
        self.output.init(init_sum_output * self.fb.infi_gain)
        self.fb.init(init_sum_output)
        r_fb = self.fb.run(r_sum[0])
        r_output = self.output.filter(r_fb[0])
        return [r_output] + r_fb + r_sum


class SumTGCTGT_T(SumTGCTGT):

    def __init__(self, parameters, gain_infi_func, gain_tau_type, dt=DT):
        self.temporal = ContainerFilter(lambda x: parameters[0], dt=dt)
        SumTGCTGT.__init__(self,
                           parameters[1:],
                           gain_infi_func,
                           gain_tau_type,
                           dt=dt)

    def run(self, init_stim, trace):
        r = SumTGCTGT.run(self, init_stim, trace)
        self.temporal.init(r[0][0])
        o = self.temporal.filter(r[0])
        return [o] + r


class SumTGCTGT_2T(SumTGCTGT_T):
    """ add a temporal for TG as one subunit is T-TG-T """

    def __init__(self, parameters, gain_infi_func, gain_tau_type, dt=DT):
        self.temporal2 = ContainerFilter(lambda x: parameters[0], dt=dt)
        SumTGCTGT_T.__init__(self,
                             parameters[1:],
                             gain_infi_func,
                             gain_tau_type,
                             dt=dt)

    def run(self, init_stim, trace):
        self.temporal2.init(init_stim)
        f1 = self.temporal2.filter(trace)
        r1 = self.tgc.run(init_stim, trace)
        r2 = self.tgt.run(init_stim, f1)
        s = r1[0] * self.w1 + r2[0] * (1 - self.w1)
        self.temporal.init(s[0])
        o = self.temporal.filter(s)
        return [o, s] + r1 + [f1] + r2


class SumTGCT(SumTGC):
    """ 2 TGCs + T """

    def __init__(self, parameters, gain_infi_func, gain_tau_type, dt=DT):
        tau = parameters[0]
        self.gain_infi_func = gain_infi_func
        self.temporal = ContainerFilter(lambda x: tau, dt=dt)
        SumTGC.__init__(self,
                        parameters[1:],
                        gain_infi_func,
                        gain_tau_type,
                        dt=dt)

    def run(self, init_stim, trace):
        res = SumTGC.run(self, init_stim, trace)
        init_o = init_stim * self.gain_infi_func(init_stim)
        self.temporal.init(init_o)
        r = self.temporal.filter(res[0])
        return [r] + res


class ShareTGC2TT:
    """ Single TGC + 2Ts + T"""

    def __init__(self, parameters, gain_infi_func, gain_tau_type, dt=DT):
        tau11 = parameters[0]
        tau12 = parameters[1]
        tau2 = parameters[2]
        self.temporal11 = ContainerFilter(lambda x: tau11, dt=dt)
        self.temporal12 = ContainerFilter(lambda x: tau12, dt=dt)
        self.temporal2 = ContainerFilter(lambda x: tau2, dt=dt)

        self.w1 = max(0, min(parameters[3], 1))
        self.gain_infi_func = gain_infi_func
        self.tgc = TGC(parameters[4:], gain_infi_func, gain_tau_type, dt=dt)

    def run(self, init_stim, trace):
        r = self.tgc.run(init_stim, trace)
        init = self.gain_infi_func(init_stim) * init_stim
        self.temporal11.init(init)
        self.temporal12.init(init)
        self.temporal2.init(init)
        t1 = self.temporal11.filter(r[0])
        t2 = self.temporal12.filter(r[0])
        r = self.temporal2.filter(t1 * self.w1 + t2 * (1 - self.w1))
        return [r, t1, t2] + r


class SumTGC2TT(SumTGC):
    """ 2 TGCs + 2Ts + T """

    def __init__(self, parameters, gain_infi_func, gain_tau_type, dt=DT):
        tau11 = parameters[0]
        tau12 = parameters[1]
        tau2 = parameters[2]
        self.temporal11 = ContainerFilter(lambda x: tau11, dt=dt)
        self.temporal12 = ContainerFilter(lambda x: tau12, dt=dt)
        self.temporal2 = ContainerFilter(lambda x: tau2, dt=dt)

        self.gain_infi_func = gain_infi_func
        w1 = parameters[3]
        if w1 < 0.:
            w1 = 0.
        elif w1 > 1.:
            w1 = 1.
        k = parameters[4]
        infi1 = lambda x: k / (x + k)

        def infi2(x):
            o1 = x * infi1(x) * w1
            expected_output = x * gain_infi_func(x)
            gain = (expected_output - o1) / (x * (1 - w1))
            return gain

        self.w1 = w1
        n = int((len(parameters) - 5) / 2)
        ps1 = parameters[5:5 + n]
        ps2 = parameters[5 + n:]
        self.tgc1 = TGC(ps1, infi1, gain_tau_type, dt=dt)
        self.tgc2 = TGC(ps2, infi2, gain_tau_type, dt=dt)
        self.infi1 = infi1
        self.infi2 = infi2

    def run(self, init_stim, trace):
        r1 = self.tgc1.run(init_stim, trace)
        r2 = self.tgc2.run(init_stim, trace)
        init_1 = self.infi1(init_stim) * init_stim
        init_2 = self.infi2(init_stim) * init_stim
        init_s = init_1 * self.w1 + init_2 * (1 - self.w1)
        self.temporal11.init(init_1)
        self.temporal12.init(init_2)
        self.temporal2.init(init_s)
        t1 = self.temporal11.filter(r1[0])
        t2 = self.temporal12.filter(r2[0])
        r = self.temporal2.filter(t1 * self.w1 + t2 * (1 - self.w1))
        return [r, t1, t2] + r1 + r2


class Sum21TGC:
    """ Two subunits: 2TGCS + 1 TGC"""

    def __init__(self, parameters, gain_infi_func, gain_tau_type, dt=DT):
        w1 = parameters[0]
        k11 = parameters[1]
        k12 = parameters[2]
        if w1 < 0.001:
            w1 = 0.001
        elif w1 > 0.999:
            w1 = 0.999
        infi11 = np.vectorize(lambda x: k11 / (x + k11))
        infi12 = np.vectorize(lambda x: k12 / (x + k12))

        def infi2(x):
            o11 = x * infi11(x)
            o12 = o11 * infi12(o11) * w1
            expected_output = x * gain_infi_func(x)
            gain = (expected_output - o12) / (x * (1 - w1))
            return gain

        self.w1 = w1
        self.infi11 = infi11
        self.infi12 = infi12
        self.infi2 = infi2
        n = int((len(parameters) - 3) / 3)
        ps1 = parameters[3:3 + n]
        ps2 = parameters[3 + n:3 + n * 2]
        ps3 = parameters[3 + n * 2:]
        self.tgc11 = TGC(ps1, infi11, gain_tau_type, dt=dt)
        self.tgc12 = TGC(ps2, infi12, gain_tau_type, dt=dt)
        self.tgc2 = TGC(ps3, infi2, gain_tau_type, dt=dt)

    def run(self, init_stim, trace):
        r11 = self.tgc11.run(init_stim, trace)
        r12 = self.tgc12.run(init_stim * self.infi11(init_stim), r11[0])
        r2 = self.tgc2.run(init_stim, trace)
        r = self.w1 * r12[0] + (1 - self.w1) * r2[0]
        return [r] + r2 + r11 + r12


class Sum2TGC1TGC:
    """ Two layers: 2 TGCS + TGC """

    def __init__(self,
                 parameters,
                 gain_infi_func,
                 gain_tau_type,
                 dt=DT,
                 max_x=100):
        w1 = parameters[0]
        k1 = parameters[1]
        k2 = parameters[2]
        if w1 < 0.001:
            w1 = 0.001
        elif w1 > 0.999:
            w1 = 0.999

        infi1 = np.vectorize(lambda x: k1 / (x + k1))
        infi2 = np.vectorize(lambda x: k2 / (x + k2))
        xs = np.arange(0, max_x, 0.001)
        ys = w1 * xs * infi1(xs) + (1 - w1) * xs * infi2(xs)
        from scipy.interpolate import interp1d
        layer1_output_reverse_func = interp1d(ys, xs, kind="cubic")
        self.layer1_output_reverse_func = layer1_output_reverse_func

        def infi3(first_layer_output):
            if first_layer_output < ys[0]:
                first_layer_output = ys[0]
            elif first_layer_output > ys[-1]:
                first_layer_output = ys[-1]
            origin_input = layer1_output_reverse_func(first_layer_output)
            expected_output = origin_input * gain_infi_func(origin_input)
            gain = expected_output / first_layer_output
            return gain

        self.w1 = w1
        self.infi1 = infi1
        self.infi2 = infi2
        n = int((len(parameters) - 3) / 3)
        ps1 = parameters[3:3 + n]
        ps2 = parameters[3 + n:3 + n * 2]
        ps3 = parameters[3 + n * 2:]
        self.tgc11 = TGC(ps1, infi1, gain_tau_type, dt=dt)
        self.tgc12 = TGC(ps2, infi2, gain_tau_type, dt=dt)
        self.tgc2 = TGC(ps3, infi3, gain_tau_type, dt=dt)

    def run(self, init_stim, trace):
        r11 = self.tgc11.run(init_stim, trace)
        r12 = self.tgc12.run(init_stim, trace)
        r1 = self.w1 * r11[0] + (1 - self.w1) * r12[0]
        init_r1 = self.w1 * init_stim * self.infi1(init_stim)
        init_r1 += (1 - self.w1) * init_stim * self.infi2(init_stim)
        r2 = self.tgc2.run(init_r1, r1)
        return r2 + [r1] + r11 + r12


"""import photoreceptor as PR
PR_INFI = lambda x: PR.DA14_Cone.alpha*x*100/(1+PR.DA14_Cone.beta*x*100)

BC_INFI_FUNC = lambda u: 1/((44./65)+(1+(44./37))*u)
PRBC_INFI_FUNC = lambda u: u*BC_INFI_FUNC(u)/PR_INFI(u)

AC_INFI_FUNC = lambda u: 1/(7/43.+.4*7/(.01*6)+(1.+7./6)*u)
ONGC_INFI_FUNC = lambda u: 1/(25/39.+(1+25/.2+25*.6/(.2*.001))*u)
OFFGC_INFI_FUNC = lambda u: 1/(19/60.+.2*19/(.02*6)+(1.+19./6)*u)"""


def fit_lnk_bc(MODEL=TGC,
               x0=[10, 20],
               gain_tau_type="constant",
               n_pool=2,
               dt=DT):
    import lnk
    import optimize as O
    bc = lnk.Kinetics()
    pre_time = 1000
    stage_time = 200
    data = dill.load(open("../datas/NEW/bc_valid.pkl"))
    # traces = O.get_step_stimulus(pre_time, stage_time, dt=dt)
    return O.optimize(MODEL,
                      x0,
                      data,
                      n_pool,
                      dt, {
                          "gain_infi_func": BC_INFI_FUNC,
                          "gain_tau_type": gain_tau_type
                      },
                      saved_pre=10)


def fit_lnk_bc_contrast(MODEL, x0, gain_tau_type="rhill", n_pool=True, dt=DT):
    import optimize as O
    import dill
    pre_time = 1000
    bc_data = dill.load(open("../datas/NEW/bc_wn_contrast.pkl", "r"))
    if MODEL == TGCT_Free:
        if x0 is None:
            x0 = np.random.random(6) * np.array([5, 5, 200, 5, 200, 200])
            print(x0)
        else:
            x0 = np.array(x0)
        while True:
            p = O.optimize_traces(MODEL,
                                  x0,
                                  traces,
                                  r_bc,
                                  pre_time,
                                  1,
                                  DT, {
                                      "n_params4infi": 2,
                                      "gain_infi_type": "rhill",
                                      "gain_tau_type": "rhill"
                                  },
                                  record_fname=fname)
            p = p['x']
            if np.abs(p - x0).sum() < 1e-6:
                return p
            x0 = p
    else:
        x0 = np.array(x0)
        while True:
            p = O.optimize(MODEL, x0, bc_data, n_pool, dt, {
                "gain_infi_func": BC_INFI_FUNC,
                "gain_tau_type": gain_tau_type
            })
            if np.abs(p['x'] - x0).sum() < 1e-6:
                return p
            x0 = p['x']


def fit_lnk_bc_pr(MODEL, x0, n_pool, dt=DT):
    import dill
    import optimize as O
    data = dill.load(open("../datas/NEW/bc_valid_pr.pkl"))
    pre_time, inputs, outputs = tuple(data)
    inputs = [i + 45 for i in inputs]
    return O.optimize_traces(MODEL, x0, inputs, outputs, pre_time, n_pool, dt, {
        "gain_infi_func": PRBC_INFI_FUNC,
        "gain_tau_type": gain_tau_type
    })


def fit_lnk_bc_Franke(x0=None, MODEL=TGCT_Free, record_fname=None):
    import optimize as O
    pre_time = 2000
    stim = np.loadtxt('../datas/Franke_stim.txt')
    resp = np.loadtxt('../datas/Franke_resp_kbc.txt')[pre_time * 10:]
    print(len(stim), len(resp))
    if MODEL == TGCT_Free:
        if x0 is None:
            x0 = np.random.random(6) * np.array([5, 5, 200, 5, 200, 200])
        else:
            x0 = np.array(x0)
        while True:
            p = O.optimize_traces(MODEL,
                                  x0, [stim], [resp],
                                  pre_time,
                                  1,
                                  DT, {
                                      "n_params4infi": 2,
                                      "gain_infi_type": "rhill",
                                      "gain_tau_type": "rhill"
                                  },
                                  record_fname=record_fname)
            p = p['x']
            if np.abs(p - x0).sum() < 1e-6:
                return p
            x0 = p
    else:
        return O.optimize_traces(MODEL,
                                 x0, [stim], [resp],
                                 pre_time,
                                 1,
                                 DT, {
                                     "gain_infi_func": BC_INFI_FUNC,
                                     "gain_tau_type": "rhill"
                                 },
                                 record_fname=record_fname)


"""
Gold-standard: 
    TGCT([ 1.62237667,  0.49689773, 22.73320449, 16.94862102], M.BC_INFI_FUNC, "rhill"), 195.46631969087665
search with inital parameters:
    # init: [1/((1+44/37.)*65/44), 65/44., 0.61515855,  0.46109641, 24.67115539, 18.08443383]
    [ 0.30921178,  1.47727272,  1.57064778,  0.48663334, 23.12956289, 17.06654208], 195.5414733414584
search totally free:
    [0.30919492, 1.47733815, 1.57756141, 0.49236695, 22.87908451, 16.99652194], 195.4786261471294
    [0.30921177,  1.47727273,  1.57166673,  0.48552017, 23.14603501, 17.05245571], 195.5614073335579
    [ 0.30921177,  1.47727273,  1.63614687,  0.49891112, 22.67318222,16.93494613], 195.47307963228155
    [ 0.30921178,  1.47727273,  1.59785315,  0.49948349, 22.68277584, 16.9995148 ], 195.48534903161607
    [0.226486333, 1.92303658, 0.0499999996, 0.213662539, 52.0190702, 21.7361031], 947.232064996921823

"""


def fit_lnk_ac(MODEL=TGC,
               x0=[10, 20],
               gain_tau_type="constant",
               n_pool=10,
               record_fname=None,
               ext_parameters={}):
    import optimize as O
    dt = 0.1
    saved_pre = 10
    pre_time = 90000
    stage_time = 30000
    data = list(dill.load(open("../datas/NEW/ac_valid.pkl")))
    n_stage = int(stage_time / dt)
    n_pre = n_stage + int(saved_pre / dt)
    data[1] = [i[:n_pre] for i in data[1]]
    data[2] = [i[:n_stage] for i in data[2]]
    ext_parameters.update({
        "gain_infi_func": AC_INFI_FUNC,
        "gain_tau_type": gain_tau_type
    })
    while True:
        p = O.optimize(MODEL,
                       x0,
                       data,
                       n_pool,
                       dt,
                       ext_parameters,
                       saved_pre=saved_pre,
                       evaluate_method=O.evaluate_abs,
                       record_fname=record_fname)
        if np.abs(p['x'] - x0).sum() < 1e-6:
            return p
        x0 = p['x']


def fit_lnk_ongc(MODEL=TGC,
                 x0=[10, 20],
                 gain_tau_type="constant",
                 n_pool=10,
                 dt=DT):
    import optimize as O
    pre_time = 20000
    stage_time = 2000
    data = dill.load(open("../datas/NEW/ongc_valid.pkl"))
    x0 = np.array(x0)
    while True:
        p = O.optimize(MODEL,
                       x0,
                       data,
                       n_pool,
                       dt, {
                           "gain_infi_func": ONGC_INFI_FUNC,
                           "gain_tau_type": gain_tau_type
                       },
                       saved_pre=10)
        if np.abs(p['x'] - x0).sum() < 1e-6:
            return p
        x0 = p['x']


def fit_lnk_offgc(MODEL=TGC, x0=[10, 20], gain_tau_type="constant", n_pool=10):
    import optimize as O
    dt = 0.1
    pre_time = 100000
    stage_time = 50000
    data = dill.load(open("../datas/NEW/offgc_valid.pkl"))
    # traces = O.get_step_stimulus(pre_time, stage_time, dt=dt)
    x0 = np.array(x0)
    while True:
        p = O.optimize(MODEL,
                       x0,
                       data,
                       n_pool,
                       dt, {
                           "gain_infi_func": OFFGC_INFI_FUNC,
                           "gain_tau_type": gain_tau_type
                       },
                       saved_pre=10)
        if np.abs(p['x'] - x0).sum() < 1e-6:
            return p
        x0 = p['x']


def fit_glm_slow(MODEL=GC, x0=[20], gain_tau_type="constant", n_pool=10, dt=DT):
    import glm
    import optimize as O
    slow_glm = glm.slow_glm
    pre_time = 250
    stage_time = 250
    traces = O.get_step_stimulus(pre_time, stage_time, dt=dt)
    glm_output_func = slow_glm.get_stable_output_func()
    gain_infi_func = lambda x: glm_output_func(x) / x
    return O.optimize(MODEL, x0, slow_glm, traces, pre_time, n_pool, dt, {
        "gain_infi_func": gain_infi_func,
        "gain_tau_type": gain_tau_type
    })


"""
Usage:

import dill
import matplotlib.pyplot as plt
import models as M


# --------------------------------------------------
# for AC
p = [.... parameters ....]
# p = [-6.57752011e-02,  3.03833127e+00,  1.99813159e+01,  2.17166805e+01,
        9.43300341e+02,  5.00050835e-02,  3.90656831e+03,  4.80088428e+03,
        5.50147176e+01]

# check manually
model = M.Sum2TGC1TGC(p, M.AC_INFI_FUNC, "constant")
data = dill.load(open("../datas/NEW/ac_valid.pkl"))
inits, stims, resps = tuple(data)
resp_model = model.run(inits[2], stims[2])  # #2 trace
plt.plot(resp_model[0][100:]) # response of model
plt.plot(resps[2])  # response in data

[9.120e-01, 3.038e+03, 3.038e+00, 5.171e+03, 9.430e+02, 6.170e+00,
       6.430e+01, 4.800e-01, 5.500e+01]), 3842.2643104195918
[1.51757801e-01, 8.02366908e+03, 1.48130651e-01, 4.41266191e+03,
        1.59472876e+03, 6.18506772e-02, 6.50598921e-07, 1.21244175e+00,
        7.19517087e+01], 2572.7484485163036


# stage_time = 2000
[ 7.27320111e-02, -3.63495482e+04,  1.48754022e-01,  3.80607654e+03,
        6.91175110e-12,  5.53976099e+00,  1.39575695e-05,  5.06960753e-02,
        8.03721765e+02]

# optimize automatically
n_pool = 5 # number of processes in parallel running
M.fit_lnk_ac(M.Sum2TGC1TGC, p, "constant", n_pool)

# --------------------------------------------------
# for OFFGC
p = [.... parameters ....]
# p = [ 7.37137810e-02,  1.33896909e+01, -1.99516660e-04,  1.80017547e+01,
        2.21537966e+02,  5.81727139e+01,  6.55776256e-02,  1.45739063e+03,
        1.90107333e+03]

# check manually
model = M.Sum2TGC1TGC(p, M.OFFGC_INFI_FUNC, "constant")
data = dill.load(open("../datas/NEW/offgc_valid.pkl"))
inits, stims, resps = tuple(data)
resp_model = model.run(inits[2], stims[2]) # #2 trace 
plt.plot(resp_model[0]) # response of model
plt.plot(resps[2])  # response in data

# optimize automatically
n_pool = 5 # number of processes in parallel running
M.fit_lnk_offgc(M.Sum2TGC1TGC, p, "constant", n_pool)





LNK_BC:
TGC([9.4956689 , 2.82397544], BC_INFI_FUNC, "constant", DT), 731.310934296575
TGC([12.23484523,  0.37063074, 23.8356509 ], BC_INFI_FUNC, "rhill", DT), 713.0312457844138
TGCTGC([ 0.80036409,  5.88661299,  1.37740411,  4.3438891 , 12.46951168], 
       BC_INFI_FUNC, "constant", DT), 281.8752730328775
TGCTGC([ 0.58845018,  0.58412144,  0.44919149, 31.52720272, 19.37708175,
        0.54015695, 27.34654094], BC_INFI_FUNC, "rhill", DT), 71.01638698096666
SumTGCT([ 2.48880564,  0.4721047 ,  0.15158959, 13.3269539 ,  8.2442831 ,
        5.36790699,  1.0859184 , 45.83442577], BC_INFI_FUNC, "constant", DT)), 335.0355909557312

TGCT([ 0.61515855,  0.46109641, 24.67115539, 18.08443383], BC_INFI_FUNC, "rhill", DT), 214.31976567285514
SumTGCT([19.1440723, 0.00696873345, 4.39154272, 
         26.7646439, 0.00103794244, 3.0500927e-06, 
         0.386299127, 0.380227122, 31.4322905], BC_INFI_FUNC, "rhill", DT), 181.58863529741012 # v
# r
[0.99994009, 0.999882  , 0.99976497, 0.99954862, 0.99932443,
       0.9984087 , 0.99745127, 0.99630319, 0.99636592, 0.99531127,
       0.99344484]

LNK_AC:
TGC([4780.15454782, 3640.13437625], AC_INFI_FUNC, "constant", DT), 10728
TGC([4.78016942e+03, 9.87102054e+08, 3.64026584e+03], AC_INFI_FUNC, "rhill", DT), 10728.839738808649
SumTGC([4.86877396e-03, 1.57900407e+00, 2.45544639e+00, 6.35513853e-03,
       4.74751246e+03, 4.36888735e+03],  AC_INFI_FUNC, "constant", DT), 9046.09557923422
SumTGC([  3.06110617e-03,   1.70993912e+00,   2.68108542e+00,
         3.52115853e-08,   4.05757827e-01,   4.78376991e+03,
         7.20088600e+00,   6.74102021e+03], AC_INFI_FUNC, "rhill", DT), 8886.9779566435882

# SumTGC_FB: # 8894.6151102280601
[  1.97172608e+00,  -2.66214817e+00,  -4.11183006e+01,
         1.12303093e-02,   1.69016669e+00,   4.74742332e+00,
         8.92709848e-03,   9.26347605e+03,   4.78038603e+03,
         7.21710220e+00,   6.71781496e+03]), 8880.8601314270891


# Sum3TGC([-1.82485479e-01,  1.62814282e-01,  1.25526987e-01,  2.32455282e+00,
        2.15144097e+01,  3.01082081e+02,  1.04536399e+04,  1.01220982e+00,
        3.52345059e+03,  1.52877077e+03]
# Sum21TGC [2.17073436e-01, 1.38981427e+00, 1.93147436e+00, 8.35891893e+02,
       1.86692263e+03, 4.32075465e+03, 1.00052181e+03, 2.48643609e+02,
       1.88324137e+02]

# Sum2TGC1TGC [-6.57752011e-02,  3.03833127e+00,  1.99813159e+01,  2.17166805e+01,
        9.43300341e+02,  5.00050835e-02,  3.90656831e+03,  4.80088428e+03,
        5.50147176e+01]), 9565.63069709991
 
LNK_ONGC:
TGC([58.26105885, 21.01361902], ONGC_INFI_FUNC, "constant", DT), 0.5689837852543299
TGC([46.7815620, 0.00323720508, 7732.60942], ONGC_INFI_FUNC, "rhill", DT), 0.381286182809011
TGCTGC  [0.72611683, 18.40028405,  2.74607555,  3.61969344, 49.98722401], "constant", 0.4522778676091192
TGCTGC( [7.90916868e-01, 1.46257676e+01, 6.60741400e-08, 1.41154001e-09,
       1.66617940e+00, 4.29496511e-04, 6.35682078e+04], ONGC_INFI_FUNC, "rhill", DT), 0.19092128366796834 # o
r: [0.72177028, 0.61382025, 0.62696536, 0.8219874 , 0.92350596,
       0.72883536, 0.97372623, 0.97223838, 0.98420892, 0.97702886,
       0.99163295]

SumTGC([ 0.1656852 ,  1.09096699, 74.76194791, 34.61340234, 74.76462768,
       34.61845147], ONGC_INFI_FUNC, "constant", DT), 0.582983283095631
# TGCTGCTGC #o
      [ 1.11020821e+00,  2.37808065e+00,  1.31259051e+01,  9.60056935e+00,
        3.46785156e+00,  5.34766396e+00, -8.95900662e-01,  5.77224304e-21,
        7.85492320e-02,  6.60893943e-04,  4.13371651e+04]), 0.1874781077237118
r: [0.79040966, 0.73951606, 0.78215069, 0.64797637, 0.63900956,
       0.81609057, 0.97769346, 0.98642957, 0.98921677, 0.99005109,
       0.99368795]


# Sum21TGC [-2.35364284e+00,  6.86199535e+00,  9.32119715e+00,  6.77636472e+01,
        5.22506751e-02,  7.68653725e-12,  5.29953301e-02,  3.18471423e-04,
        4.62235097e+02,  7.39174241e+01,  2.15253606e-01,  1.50804830e+02]), 0.4376521807605286

LNK_OFFGC:
TGC([6071.66724357, 3098.31874266], OFFGC_INFI_FUNC, "constant", DT), 22918.50687773356
TGC([6.07166914e+03, 2.60241815e+09, 3.09832410e+03], OFFGC_INFI_FUNC, "rhill", DT), 22918.506879753935
SumTGC([4.66832413e-03, 1.53024850e+00, 1.10485719e+01, 3.17519115e+01,
       4.74212597e+03, 4.31390162e+03], OFFGC_INFI_FUNC, "constant", DT), 16971.152948412546
Sum3TGC() # [7.90899413e-01, 2.19974406e-02, 6.28615815e-01, 1.24677541e+00,
       5.45790770e+03, 1.49776259e+03, 7.36969148e+00, 4.37004767e+01,
       5.31842278e+03, 1.49171055e+03], 12768.783275844948
# [ 7.37137810e-02,  1.33896909e+01, -1.99516660e-04,  1.80017547e+01,
        2.21537966e+02,  5.81727139e+01,  6.55776256e-02,  1.45739063e+03,
        1.90107333e+03]), 64747.488452052705

SumTGC_FB
[ -8.77426343e+00,  -2.42814254e+00,   5.39056623e-01,
         3.55715867e-02,   1.18652438e+00,   4.47492447e+00,
         1.25669879e+00,   4.77476516e-03,   6.47832018e+03,
         3.10866674e+00,   1.02797884e+04]), 16052.859297557865
"""
