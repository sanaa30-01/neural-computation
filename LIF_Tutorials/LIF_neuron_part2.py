#Tutorial 2: introducing spikes in LIF neuron + working with effect of the refractory period on spiking dynamics

import numpy as np
import matplotlib.pyplot as plt

t_max = 150e-3  #s
dt = 1e-3  #s
tau = 20e-3  #s
el = -60e-3  #mV
vr = -70e-3  #mV
vth = -50e-3  #mV
r = 100e6  #ohm
i_mean = 25e-11  #A 

#coding exercise 1: plotting a histogram
#aim: plot a histogram of J = 50 bins and N = 10000 realizations of V(t) for t = t_max/10 and t = t_max
#this histogram is not about progression over time, it is about the distribution of voltages at a specific time point

np.random.seed(2020)

#creating a time range from 0 to t_max with step size dt
t_range = np.arange(0, t_max, dt)

#calculating the number of steps by taking the length of the time range
step_end = len(t_range)
n = 10000
v_n = el * np.ones([n, step_end])

random_num = 2 * np.random.random(size=[n, step_end]) - 1
i = i_mean * (1 + 0.1 * (t_max/dt) ** 0.5 * random_num)

nbins = 50

for step, t in enumerate(t_range):

    if step == 0:
        continue

    v_n[:, step] = v_n[:, step - 1] + dt/tau * (el - v_n[:, step - 1] + r * i[:, step])

with plt.xkcd():
    plt.figure()
    plt.ylabel("Frequency")
    plt.xlabel("Membrane Potential (V)")

    #plotting histogram of voltages of all neurons at t = t_max/10
    #int(step_end/10) is the index of the time point t = t_max/10
    #int(step_end/10) is used because step_end/10 = 15 which is the column for 0.015s which is t_max/10
    plt.hist(v_n[:, int(step_end/10)], bins = nbins, histtype = 'stepfilled', linewidth = 0, label = f't = {t_max/10} s')

    #plotting histogram of voltages of all neurons at t = t_max
    #-1 is the last column which is the column for t = t_max
    plt.hist(v_n[:, -1], bins = nbins, histtype = 'stepfilled', linewidth = 0, label = f't = {t_max} s')

    plt.legend()
    plt.show()