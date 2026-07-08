#Tutorial 2: introducing spikes in LIF neuron + working with effect of the refractory period on spiking dynamics

import numpy as np
import matplotlib.pyplot as plt


def plot_all(t_range, v, raster=None, spikes=None, spikes_mean=None):
    """
    Plots Time evolution for
    (1) multiple realizations of membrane potential
    (2) spikes
    (3) mean spike rate (optional)

    Args:
        t_range (numpy array of floats)
            range of time steps for the plots of shape (time steps)

        v (numpy array of floats)
            membrane potential values of shape (neurons, time steps)

        raster (numpy array of floats)
            spike raster of shape (neurons, time steps)

        spikes (dictionary of lists)
            list with spike times indexed by neuron number

        spikes_mean (numpy array of floats)
            Mean spike rate for spikes as dictionary

    Returns:
        Nothing.
    """
    #calculating the mean voltage of all neurons at each time step
    n = v.shape[0]
    v_mean = np.mean(v, axis=0)
    fig_w, fig_h = plt.rcParams['figure.figsize']
    plt.figure(figsize=(fig_w, 1.5 * fig_h))

    #creating the first subplot for the mean voltage of all neurons
    ax1 = plt.subplot(3, 1, 1)
    #looping through the neurons --> j is the neuron index
    for j in range(n):
        plt.scatter(t_range, v[j], color="k", marker=".", alpha=0.01)
    plt.plot(t_range, v_mean, 'C1', alpha=0.8, linewidth=3)
    plt.xticks([])
    plt.ylabel(r'$V_m$ (V)')

    if raster is not None:
        #creating the second subplot for the spike raster
        plt.subplot(3, 1, 2)
        spikes_mean = np.mean(raster, axis=0)
        plt.imshow(raster, cmap='Greys', origin='lower', aspect='auto')

    else:
        #creating the second subplot for the spikes of all neurons
        plt.subplot(3, 1, 2, sharex=ax1)
        #looping through neurons to collect the spike times for each neuron and plot them as scatter points at height j
        for j in range(n):
            #converting the list of spike times to a numpy array --> spikes[j] is the dictionary of spike times for the neuron j
            times = np.array(spikes[j])
            #plotting the spike times as scatter points at height j --> j*np.ones_like(times) is the height of the scatter points
            plt.scatter(times, j * np.ones_like(times), color="C0", marker=".", alpha=0.2)

    plt.xticks([])
    plt.ylabel('neuron')

    if spikes_mean is not None:
        #creating the third subplot for the mean spike rate of all neurons
        plt.subplot(3, 1, 3, sharex=ax1)
        #plotting the mean spike rate as a line plot
        plt.plot(t_range, spikes_mean)
        plt.xlabel('time (s)')
        plt.ylabel('rate (Hz)')

    #tightly layout the subplots to prevent overlap
    plt.tight_layout()
    plt.show()


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


#coding exercise 2: adding spiking to the LIF neuron

np.random.seed(2020)

t_range = np.arange(0, t_max, dt)
step_end = len(t_range)
n = 500
v_n = el * np.ones([n, step_end])

random_num = 2 * np.random.random(size=[n, step_end]) - 1
i = i_mean * (1 + 0.1 * (t_max/dt) ** 0.5 * random_num)

#intiializing the spikes dictionary --> will be used to store the spikes of each neuron
#the key is the neuron index and the value is a list of spike times
spikes = {j: [] for j in range(n)}

#intializing the spikes_n array --> will be used to store the number of spikes at each time step
#this array will be used to calculate the mean spike rate
spikes_n = np.zeros([step_end])

#looping through the time steps by using enumerate --> step is the time step index and t is the time
for step, t in enumerate(t_range):
    if step == 0:
        continue

    v_n[:, step] = v_n[:, step - 1] + dt/tau * (el - v_n[:, step - 1] + r * i[:, step])

    #looping through the neurons by using range(n) --> j is the neuron index
    for j in range(n):

        #checking if the voltage of the neuron is greater than the threshold to check if the neuron has spiked
        #using the voltage at the current time step to check if the neuron has spiked --> j is the neuron index and step is the time step index
        if v_n[j, step] >= vth:

            #resetting the voltage of the neuron to the resting potential 
            v_n[j, step] = vr

            #adding the spike time to the spikes dictionary --> j is the neuron index and t is the spike time
            #use [t] and not [step] because step is the time step index and t is the time and we want to store the spike time not the time step index in the dictionary
            spikes[j] += [t]

            #incrementing the number of spikes at the current time step 
            #[step] is the time step index and we use this to store the number of spikes at the current time step in the spikes_n array
            spikes_n[step] += 1

#calculating the mean spike rate by dividing the number of spikes at each time step by the number of neurons
spikes_mean = spikes_n/n

#calculating the mean voltage of all neurons at each time step
v_mean = np.mean(v_n, axis=0)

with plt.xkcd():
    plt.figure()

    #creating the first subplot for the mean voltage of all neurons
    ax1 = plt.subplot(3, 1, 1)
    #looping through the neurons --> j is the neuron index
    for j in range(n):
        plt.scatter(t_range, v_n[j], color='k', marker='.', alpha=0.01)
    plt.plot(t_range, v_mean, color='C1', alpha=0.8, linewidth=3)
    plt.ylabel("V_m (V)")

    #creating the second subplot for the spikes of all neurons
    plt.subplot(3, 1, 2, sharex=ax1)
    #looping through neurons to collect the spike times for each neuron and plot them as scatter points at height j
    for j in range(n):
        #converting the list of spike times to a numpy array --> spikes[j] is the dictionary of spike times for the neuron j
        times = np.array(spikes[j])
        #plotting the spike times as scatter points at height j --> j*np.ones_like(times) is the height of the scatter points
        plt.scatter(times, j*np.ones_like(times), color='C0', marker='.', alpha=0.2)

    plt.ylabel("Neurons")

    #creating the third subplot for the mean spike rate of all neurons
    plt.subplot(3, 1, 3, sharex=ax1)
    #plotting the mean spike rate as a line plot
    plt.plot(t_range, spikes_mean)
    plt.xlabel("Time(s)")
    plt.ylabel("rate(Hz)")

    #tightly layout the subplots to prevent overlap
    plt.tight_layout()
    plt.show()


#coding exercise 3: using boolean indexing 
#can avoid looping through all neurons in each time step by identifying the indexes of the neurons that spiked in the previous step 

np.random.seed(2020)

t_range = np.arange(0, t_max, dt)
step_end = len(t_range)
n = 500
v_n = el * np.ones([n, step_end])

random_num = 2 * np.random.random(size=[n, step_end]) - 1
i = i_mean * (1 + 0.1 * (t_max/dt) ** 0.5 * random_num)

spikes = {j: [] for j in range(n)}
spikes_n = np.zeros([step_end])

for step, t in enumerate(t_range):
    if step == 0:
        continue

    v_n[:, step] = v_n[:, step - 1] + dt/tau * (el - v_n[:, step - 1] + r * i[:, step])

    #using boolean indexing to identify the neurons that have spiked
    #v_n[:, step] is the voltage of all neurons at the current time step and we are checking if it is greater than the threshold
    spiked = (v_n[:, step] >= vth)
    
    #resetting the voltage of the neurons that have spiked to the resting potential
    v_n[spiked, step] = vr

    #looping through the neurons that have spiked using np.where(spiked)[0] --> this returns the indexes of the neurons that have spiked
    for j in np.where(spiked)[0]:
        spikes[j] += [t]
        spikes_n[step] += 1

spikes_mean = spikes_n/n

# Plot multiple realizations of Vm, spikes and mean spike rate
plot_all(t_range, v_n, spikes=spikes, spikes_mean=spikes_mean)


#coding exercise 4: making a binary raster plot
#aim: binary raster plot represents spike times as 1s in a binary grid initialized to 0s 

np.random.seed(2020)

t_range = np.arange(0, t_max, dt)
step_end = len(t_range)
n = 500
v_n = el * np.ones([n, step_end])

random_num = 2 * np.random.random(size=[n, step_end]) - 1
i = i_mean * (1 + 0.1 * (t_max/dt) ** 0.5 * random_num)

#intializing the raster array --> will be used to store the binary raster plot
raster = np.zeros([n, step_end])

for step, t in enumerate(t_range):
    if step == 0:
        continue

    v_n[:, step] = v_n[:, step - 1] + (dt/tau) * (el - v_n[:, step - 1] + r * i[:, step])

    spiked = v_n[:, step] >= vth
    v_n[spiked, step] = vr 
    #setting the raster array to 1 for the neurons that have spiked
    #removed the loop using np.where to avoid needing 2 arrays --> raster and spiked
    raster[spiked, step] = 1    # --> raster[spiked, step] = 1 is equivalent to raster[np.where(spiked)[0], step] = 1

plot_all(t_range, v_n, raster=raster)


#coding exercise 5: investigating refractory periods

np.random.seed(2020)

t_range = np.arange(0, t_max, dt)
step_end = len(t_range)
n = 500
v_n = el * np.ones([n, step_end])

random_num = 2 * np.random.random(size=[n, step_end]) - 1
i = i_mean * (1 + 0.1 * (t_max/dt) ** 0.5 * random_num)

raster = np.zeros([n, step_end])

#initializing the refractory period as 10 ms --> this is the time after a spike that the neuron cannot spike again
t_ref = 0.01

#initializing the last spike time as -t_ref --> this is the time of the latest spike for each neuron
#at the beginning, no neuron has spiked yet --> we still need a value for every neuron --> used -t_ref to avoid making neurons already start in refractory period
#pretending the last spike was exactly one refractory period before t = 0
last_spike = -t_ref * np.ones([n])

for step, t in enumerate(t_range):
    if step == 0:
        continue

    v_n[:, step] = v_n[:, step - 1] + (dt/tau) * (el - v_n[:, step - 1] + r * i[:, step])

    spiked = v_n[:, step] >= vth
    v_n[spiked, step] = vr
    raster[spiked, step] = 1

    #clamping the voltage of the neurons in refractory period so that they cannot spike again
    #t - last_spike is the time since the last spike --> if it is less than t_ref, the neuron is in refractory period
    clamped = (t - last_spike < t_ref)
    #setting the voltage of the neurons in refractory period to the resting potential
    v_n[clamped, step] = vr

    #updating the last spike time for the neurons that are spiking now --> t is the current time step
    last_spike[spiked] = t

plot_all(t_range, v_n, raster=raster)