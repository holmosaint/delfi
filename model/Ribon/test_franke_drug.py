import numpy as np

import Franke as F

x0 = [2.27633865055, 1.22553023225, 1.07498915909, 0.0937287922536, 1, 0.283049219763, -0.00259711606511, 11.9417995412, 3.10194014885, 0.0652735274011, -0.00197787821882, 2.64593209617, 1.09291600372, 0.181774664845, 1, 0.814659087602, 0.567507584994, 561.796112866]
x0 = np.array(x0)

x0_base = x0*0.1
x0_upper = x0*10

x0_upper[7] = 1500
x0_base[7] = 100
x0_upper[-1] = 1500
x0_base[-1] = 100

DT = 2
ext_parameters = {}
MODEL = F.SinglePathwayModel

# do not change it
CLUSTER = 7
data = F.get_data_pair_lchirp(CLUSTER, DATATYPE=0, RESPTYPE=3, dt=DT)
pre, stim, resp = tuple(data)


