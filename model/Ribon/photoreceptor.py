#coding:UTF-8

import math
import numpy as np

from .configures import *

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

class DA13:
    """ (Clark et al, 2013, pcb) """
    def __init__(self, tau_r, alpha,
                 beta, n_y, tau_y, n_z, tau_z, gamma,
                 dt=DT):
        self.dt = dt
        buffer = max(tau_y, tau_z)*10
        self.buffer = np.zeros(int(buffer/dt))
        self.alpha = alpha
        self.beta = beta
        self.tau_r = tau_r
        
        x = np.arange(self.buffer.shape[0])*dt
        self.k_y = np.vectorize(lambda x: x**(n_y)*np.exp(-x/float(tau_y))/(math.gamma(n_y+1)*tau_y**(n_y+1)))
        self.y_weights = self.k_y(x) 
        self.y_weights /= sum(self.y_weights)
        self.k_z = np.vectorize(lambda x: x**(n_z)*np.exp(-x/float(tau_z))/(math.gamma(n_z+1)*tau_z**(n_z+1)))
        self.z_weights = (self.y_weights*gamma+self.k_z(x)*(1-gamma))
        self.z_weights /= sum(self.z_weights)

        r_variables = {
            "r": 0,
        }
        self.r_container = Container(r_variables, self.r_step)
        
    def r_step(self, variables, parameters):
        self.y = sum(self.y_weights*self.buffer)
        self.z = sum(self.z_weights*self.buffer)
        self.r = variables["r"]
        self.r_infi = self.alpha*self.y/(1+self.beta*self.z)
        self.tau = max(self.tau_r/(1+self.beta*self.z), self.dt)
        self.d = (-self.r+self.r_infi)/self.tau
        variables["r"] += self.dt*self.d
        return variables
            
    def refresh_buffer(self, stim):
        self.buffer[1:] = self.buffer[:-1]
        self.buffer[0] = stim

    def run(self, trace, v_dark=-45, recordings=[]):
        response = []
        records = [list() for i in recordings]
        for stim in trace:
            self.refresh_buffer(stim)
            self.r_container.step()
            response.append(self.r_container.get_variable("r")+v_dark)
            for i, item in enumerate(recordings):
                records[i].append(getattr(self, item))
        records = [np.array(i) for i in records]
        return np.array(response), records

    def reset(self):
        self.buffer[:] = 0
        self.r_container.set_variable("r", 0)

class DA14(DA13):
    """ (Szikra et al, 2014, nn) """
    def __init__(self, tau_r, alpha, beta, tau_y, tau_z, gamma, buffer=None,
                 dt=DT):
        self.dt = dt
        if buffer is None:
            buffer = max(tau_y, tau_z)*10
        self.buffer = np.zeros(int(buffer/dt))
        self.alpha = alpha
        self.beta = beta
        self.tau_r = tau_r
        
        x = np.arange(self.buffer.shape[0])*dt
        self.k_y = np.vectorize(lambda x: x*np.exp(-x/tau_y)/(tau_y**2))
        self.y_weights = self.k_y(x)
        self.y_weights /= sum(self.y_weights)
        self.k_z = np.vectorize(lambda x: x*np.exp(-x/tau_z)/(tau_z**2))
        self.z_weights = (self.y_weights*gamma+self.k_z(x)*(1-gamma))
        self.z_weights /= sum(self.z_weights)

        r_variables = {
            "r": 0,
        }
        self.r_container = Container(r_variables, self.r_step)

class DA18(DA14):
    """ (Drinnenberg et al, 2018, neuron) """
    def __init__(self, alpha, beta, tau_y, tau_z, gamma, buffer=None,
                 dt=DT):
        self.dt = dt
        if buffer is None:
            buffer = max(tau_y, tau_z)*10
        self.buffer = np.zeros(int(buffer/dt))
        self.alpha = alpha
        self.beta = beta
        
        x = np.arange(self.buffer.shape[0])*dt
        self.k_y = np.vectorize(lambda x: x*np.exp(-x/tau_y)/(tau_y**2))
        self.y_weights = self.k_y(x)
        self.y_weights /= sum(self.y_weights)
        self.k_z = np.vectorize(lambda x: x*np.exp(-x/tau_z)/(tau_z**2))
        self.z_weights = (self.y_weights*gamma+self.k_z(x)*(1-gamma))
        self.z_weights /= sum(self.z_weights)
        
    def run(self, trace, v_dark=-45, recordings=[]):
        response = []
        records = [list() for i in recordings]
        for stim in trace:
            self.refresh_buffer(stim)
            self.y = sum(self.y_weights*self.buffer)
            self.z = sum(self.z_weights*self.buffer)
            self.r_infi = self.alpha*self.y/(1+self.beta*self.z)
            response.append(self.r_infi)
            for i, item in enumerate(recordings):
                records[i].append(getattr(self, item))
        records = [np.array(i) for i in records]
        return np.array(response), records
    
class DA14R(DA14):
    """ Model for Rod in (Szikra et al, 2014, nn) """
    def __init__(self, tau_r, alpha, beta, theta, tau_y, tau_w, dt=DT):
        buffer = max(tau_y, tau_w)*10
        x = np.arange(0, buffer, step=dt)
        
        self.theta = theta
        self.alpha = alpha
        self.beta = beta
        self.tau_r = tau_r
        
        self.k_w = np.vectorize(lambda x: x*np.exp(-x/tau_w)/(tau_w**2))
        self.w_weights = self.k_y(x)
        self.w_weights /= sum(self.w_weights)
        self.k_y = np.vectorize(lambda x: x*np.exp(-x/tau_y)/(tau_y**2))
        self.y_weights = self.k_y(x)
        self.y_weights /= sum(self.y_weights)
        
        r_variables = {
            "r": 0,
        }
        self.r_container = Container(r_variables, self.r_step)

    def r_step(self, variables, parameters):
        self.y = sum(self.y_weights*self.buffer)
        self.w = sum(self.w_weights*self.buffer)
        self.r = variables["r"]
        self.r_infi = self.alpha*self.y/(1+self.beta*self.y)
        tau = self.tau_r*(1+self.theta*self.w)
        self.tau = max(tau/(1+self.beta*self.y), self.dt)
        self.d = (-self.r+self.r_infi)/self.tau
        variables["r"] += self.dt*self.d
        return variables


# input unit is R/(ms*μm**2)
DA13_DN = DA13(66, 1.4, 1.4*0.074, 3.7, 18, 7.8, 13, 0.22)
DA13_B = DA13(50, 2.1, 2.1*0.067, 3, 20, 7, 20, 0.57)

# input unit is R/(ms)
DA14_Cone = DA14(24, -8.8e-3, 1.2e-2, 43, 450, 0.64)
DA14_Rod = DA14(600, -3.9, 3.9, 2.9e-2, 17, 260)
DA18_Cone = DA18(-9.602e-6, -1.148e-5, 50.6, 576.9, 0.764)

def get_level_traces(levels, test_time=10, pre=0, pre_time=1000, post=0, post_time=5000, dt=DT):
    traces = []
    pre = [pre]*int(pre_time/dt)
    post = [post]*int(post_time/dt)
    t = int(test_time/dt)
    for level in levels:
        traces.append(pre+[level]*t+post)
    return traces

