import subprocess

for i in range(10):
    cmd = "python -X faulthandler Ribon_simulation.py Result/exp_{}/ Ribon_data.h5 > Result/exp_{}/log".format(i, i)
    print(cmd)
    cmd = cmd.split()
    job = subprocess.Popen(cmd)
    job.wait()
