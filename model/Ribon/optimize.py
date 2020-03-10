#coding:UTF-8

"""

"""

import numpy as np

from mpi4py import MPI
comm = MPI.COMM_WORLD


from configures import *
from utils import normalize_trace
# import noise as N

#########
#
# generate dataset for optimize
#
#########

def get_step_stimulus(pre_time=50000, stage_time=30000, stages=np.arange(0.1, 5, 0.5), dt=DT):
    t_stage = int(stage_time/dt)
    t_pre = int(pre_time/dt)
    traces = []
    for base in stages:
        for stage in stages:
            if base == stage:
                continue
            traces.append([base]*t_pre+[stage]*t_stage)
    return traces

def get_wn_contrast_stimulus(n_trace, pre_time, stage_time, stage_n, hz, dt=DT):
    traces = []
    for i_trace in range(n_trace):
        contrast = np.random.random(stage_n)
        trace = [0]*int(pre_time/dt)
        for c in contrast:
            wn = N.get_hz_noise(hz, 0, int(hz*stage_time/1000.), dt=dt)
            # print(len(wn))
            wn *= c
            trace += list(wn)
        traces.append(np.array(trace))
    return traces            

def get_noise_stimulus(mean_stim, std_stim, mean_steptime, std_steptime,
                       test_time, pre_time, pre_stim, dt=DT):
    trace = [pre_stim]*int(pre_time/dt)
    all_step = int(test_time/dt)+int(pre_time/dt)

def get_valid_dataset(traces, model, pre_time=50000, saved_pre=10, index=0,
                      dt=DT):
    t_begin = int((pre_time-saved_pre)/dt)
    init_stims = [i[t_begin-1] for i in traces]
    traces4evaluate = [i[t_begin:] for i in traces]
    responses = [model.run(i)[index][int(pre_time/dt):] for i in traces]
    return init_stims, traces4evaluate, responses

def evaluate_abs(resps, valids):
    v = np.array(valids)
    r = np.array(resps)
    r = r[:, :v.shape[1]]
    # for rank in range(5):
    #     index = np.array([i for i in range(len(r)) if i%5 == rank])
    #     print(rank, np.abs(r[index]-v[index]).sum())
    return np.abs(r-v).sum()

def evaluate_abs_normalized(resps, valids):
    v = np.array(valids)
    r = np.array(resps)
    r = r[:, :v.shape[1]]
    for i in range(r.shape[0]):
        r[i] = normalize_trace(r[i], r[i][0])
        v[i] = normalize_trace(v[i], v[i][0])
    return np.abs(r-v).sum()

def evaluate_r(resps, valids):
    from scipy.stats import pearsonr
    xs = []
    ys = []
    for resp, valid in zip(resps, valids):
        index = np.where(resp != valid)
        xs += resp[index].tolist()
        ys += valid[index].tolist()
    r, pvalue = pearsonr(xs, ys)
    return r

def evaluate_rr(resps, valids):
    r = evaluate_r(resps, valids)
    return 1-r

def f(i):
    if len(i) == 3:
        model, init_stim, trace = tuple(i)
        return model.run(init_stim, trace)[0]
    else:
        model, trace = tuple(i)
        return model.run(trace)[0]
        
def evaluate(model, valid_responses, traces4evaluates, need_init=True,
             init_stims=None, n_pool=1, erase_resting=False,
             evaluate_method=evaluate_abs, pre_time=0, evaluate_parameters=[],
             single_block=None,
             dt=DT):
    t_begin = int(pre_time/dt)
    if type(n_pool) == bool and n_pool:
        rank = comm.Get_rank()
        nprocs = comm.Get_size()
        import time
        if need_init:
            if single_block is None:
                model_resps = [model.run(init_stims[i], traces4evaluates[i])[0]
                               for i in range(len(init_stims))
                               if i%nprocs == rank]
            else:
                model_resps = [model.run(init_stims[i], traces4evaluates[i])[0]
                               for i in range(len(init_stims))
                               if i%single_block == rank%single_block]                
        else:
            if single_block is None:
                model_resps = [model.run(j)[0] for i, j in
                               enumerate(traces4evaluates)
                               if i%nprocs == rank]
            else:
                model_resps = [model.run(j)[0] for i, j in
                               enumerate(traces4evaluates)
                               if i%single_block == rank%single_block]
        if single_block is None:
            valid_resps = [j for i, j in enumerate(valid_responses)
                           if i%nprocs == rank]
        else:
            valid_resps = [j for i, j in enumerate(valid_responses)
                           if i%single_block == rank%single_block]
        if erase_resting:
            resting = model_resps[0][t_begin-1]
        else:
            resting = 0
        model_resps = [i[t_begin:]-resting for i in model_resps]
        value = evaluate_method(model_resps, valid_resps, *evaluate_parameters)
        return value
    elif n_pool < 2:
        if need_init:
            model_resps = [model.run(init_stim, trace)[0] for init_stim, trace
                           in zip(init_stims, traces4evaluates)]
        else:
            model_resps = [model.run(trace)[0] for trace in traces4evaluates]
    else:
        from pathos.multiprocessing import ProcessingPool as Pool
        pool = Pool(n_pool)
        if need_init:
            paras = zip([model]*len(init_stims), init_stims, traces4evaluates)
        else:
            paras = zip([model]*len(traces4evaluates), traces4evaluates)
        model_resps = pool.map(f, paras)
    if erase_resting:
        resting = model_resps[0][t_begin-1]
        model_resps = [i[t_begin:]-resting for i in model_resps]
    else:
        model_resps = [i[t_begin:] for i in model_resps]
    model_resps = [i.tolist() for i in model_resps]
    return evaluate_method(model_resps, valid_responses, *evaluate_parameters)

def optimize(MODEL, x0, data, n_pool=1, dt=DT, ext_model_parameters={}, need_init=True, 
             saved_pre=10, erase_resting=False,
             evaluate_method=evaluate_abs, optimize_parameters={},
             record_fname=None):
    init_stims, traces4evaluate, responses = tuple(data)
    # init_stims, traces4evaluate, responses = get_valid_dataset(traces, valid_model, pre_time, saved_pre=saved_pre, dt=dt)
    def func(x):
        model = MODEL(x, dt=dt, **ext_model_parameters)
        # try:
        v = evaluate(model, responses, traces4evaluate,
                         need_init, init_stims, n_pool, erase_resting,
                         evaluate_method, saved_pre, dt=dt)
        # except:
        #     v = 1e+9
        if type(n_pool) == bool:
            rank = comm.Get_rank()
            if rank == 0:
                print(x, v)
            comm.Barrier()
        else:
            print(x, v)
        if record_fname is not None:
            with open(record_fname, "a") as f:
                f.write("[%s] %s\n"%(" ".join([str(i) for i in x]), str(v)))
        return v
    from scipy.optimize import minimize
    base = 1e+9
    while True:
        opt = minimize(func, x0, method="Nelder-Mead", tol=1e-3, #options={"maxiter": 1},
                    **optimize_parameters)
        if (opt['fun'] - base) < 1e-8:
            return opt
        base = opt['fun']
        x0 = opt['x']

def optimize_traces(MODEL, x0, inputs, valid_resps, pre_time, n_pool=1, dt=DT,
                    ext_model_parameters={}, need_init=True,
                    erase_resting=False,
                    evaluate_method=evaluate_abs, evaluate_parameters=[],
                    optimize_parameters={}, except_func=None,
                    record_fname=None, single_block=None):
    t_begin = int(pre_time/dt)
    # print(x0)
    # print(inputs)
    # print(valid_resps)
    if need_init:
        init_stims = [i[t_begin-1] for i in inputs]
    else:
        init_stims = None
    def func(x):
        if except_func is not None and except_func(x):
            value = 1e+9
        else:
            try:
                model = MODEL(x, dt=dt, **ext_model_parameters)
            except:
                value = 1e+9
            else:
                value = evaluate(model, valid_resps, inputs, need_init,
                                 init_stims, n_pool, erase_resting,
                                 evaluate_method, pre_time,
                                 evaluate_parameters, single_block, dt=dt)
        if type(n_pool) is bool and n_pool:
            rank = comm.Get_rank()
            nprocs = comm.Get_size()
            comm.Barrier()
            value = comm.gather(value, root=0)
            comm.Barrier()
            value = comm.bcast(value, root=0)
            comm.Barrier()
            if single_block is None:
                if rank == 0:
                    value = sum(value)
                comm.Barrier()
                value = comm.bcast(value, root=0)
            else:
                assert nprocs%single_block == 0
                if rank%single_block == 0:
                    value = sum(value[rank:rank+single_block])
                comm.Barrier()
                if rank%single_block == 0:
                    for i in range(single_block):
                        comm.isend(value, dest=rank+i, tag=3)
                else:
                    value = comm.recv(source=rank-(rank%single_block), tag=3)
            comm.Barrier()
        v = float(value)
        if record_fname is None:
            print(x, v)
        else:
            comm.Barrier()
            rank = comm.Get_rank()
            nprocs = comm.Get_size()
            if single_block is not None and rank%single_block != 0:
                return v
            if type(n_pool) is bool and n_pool and single_block is None and rank != 0:
                return v
            if nprocs > 1:
                tail = record_fname.split('.')[-1]
                real_fname = record_fname.replace("."+tail,
                                                 "_%d.%s"%(rank, tail))
            else:
                real_fname = record_fname
            with open(real_fname, "a") as f:
                f.write("[%s] %s\n"%(", ".join([str(i) for i in x]), str(v)))
        return v
    from scipy.optimize import minimize
    base = 1e+9
    if type(n_pool) is bool and n_pool and single_block is not None:
        # running blocks in parallel
        # need set the same value for x0 in single block
        comm.Barrier()
        rank = comm.Get_rank()
        nprocs = comm.Get_size()
        if rank%single_block == 0:
            # root for single block
            for i in range(single_block):
                comm.isend(x0, dest=rank+i, tag=4)
        else:
            x0 = comm.recv(source=rank-(rank%single_block), tag=4)
        comm.Barrier()
    while True:
        opt = minimize(func, x0, method="Nelder-Mead", tol=1e-3, options={"adaptive": True},
                    **optimize_parameters)
        if (opt['fun'] - base) < 1e-8:
            return opt
        base = opt['fun']
        x0 = opt['x']


def get_parameters(out_file):
    txt = open(out_file).read()
    if 'yhrun' in txt:
        txt = txt.split('yhrun')[0]
    cons = txt.split('(array(')
    params, values = [], []
    for con in cons:
        if '])' not in con:
            continue
        con = con.split(']),')
        if len(con) != 2:
            continue
        param, value = tuple(con)
        value = float(value.splitlines()[0][:-1])
        param = " ".join(param[1:].splitlines())
        if 'Warning' in param:
            continue
        param = np.array(param.split(','), dtype=float)
        params.append(param)
        values.append(value)
    return params, values
    
    

def test_noise_r(model, valid_model, max_stage_time=100, last_time=50000,
                 xs=np.arange(0.1, 5, 0.1), pre_time=1000, dt=DT):
    from noise import get_noised_noise
    trace = get_noised_noise(xs, pre_time, last_time, max_stage_time, dt=dt)
    t_begin = int(pre_time/dt)
    resp = model.run(1, trace)[0][t_begin:]
    valid = valid_model.run(trace)[0][t_begin:]
    return evaluate_r([resp], [valid]), resp, valid
    
