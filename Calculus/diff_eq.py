import numpy as np
import matplotlib.pyplot as plt

#section 1
#population equation 
#the rate of change of population is proportional to the population itself

#relationship between population and rate of change of population
p = np.arange(0, 100, 0.1)

with plt.xkcd():

    dpdt = 0.3*p
    fig = plt.figure(figsize=(6, 4))
    plt.plot(p, dpdt)
    plt.xlabel("Population")
    plt.ylabel("Rate of change of population")
    plt.show()

#try to solve the differential equation using analytical integration
#exact solution: population grows/declines exponentially as a function of birth rate and time 

#will be plotting time and population instead of poopulation and rate of change of population
#after integrating the differential equation, we get the population as a function of time.
t = np.arange(0, 10, 0.1)

with plt.xkcd():

    p = np.exp(0.3 * t)

    fig = plt.figure(figsize=(6, 4))
    plt.plot(t, p)
    plt.xlabel("Time (years)")
    plt.ylabel("Population (millions)")

    plt.show()


#effect of changing parameters:
#if birth rate is negative, population will decline exponentially to 0.
#if birth rate is positive, population will grow exponentially to infinity.
#if birth rate is 0, population will remain constant because then rate of change becomes 0.


#section 2
#Leaky integrate and fire model

#LIF without input 

E_L = -75
tau_m = 10

V = np.arange(-90, 0, 1)
dV = -(V - E_L)/tau_m

with plt.xkcd():
    
    fig = plt.figure(figsize=(6, 4))
    plt.plot(V, dV)
    plt.hlines(0, min(V), max(V), colors='black', linestyles='dashed')
    plt.vlines(-75, min(dV), max(dV), colors='black', linestyles='dashed')

    plt.text(-50, 1, "Positive")
    plt.text(-50, -2, "Negative")
    plt.text(E_L, max(dV) + 1, r"$E_L$")
    plt.xlabel("Membrane Potential (mV)")
    plt.ylabel("Rate of change of membrane potential (mV/ms)")
    plt.ylim(-8, 2)

    plt.show()


#exact solution of the LIF model without input is the integral of the rate of change of membrane potential 
#if initial values of reset membrane potential are less than -75 mV, the membrane potential will grow exponentially to -75 mV because the derivative is positive.
#if initial values of reset membrane potential are greater than -75 mV, the membrane potential will decay exponentially to -75 mV because the derivative is negative.
#if initial values of reset membrane potential are equal to -75 mV, then the membrane potential will remain constant at -75 mV because the derivative is 0. 


#LIF with input

#with increasing input, dV/dt becomes bigger and less of it is less than 0 --> becomes more positive  

#plotting membrane potential as a function of time for a constant input 
#exact solution is V(t) which is being plotted here, time is changing with constant input current
dt = 0.5 
t_rest = 0

t = np.arange(0, 1000, dt)

tau_m = 10
R_m = 10 
V_reset = E_L = -75 

I = 10

V = E_L + R_m * I + (V_reset - E_L - R_m * I) * np.exp(-t/tau_m)

with plt.xkcd():

    fig = plt.figure(figsize=(6, 4))
    plt.plot(t, V)
    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane Potential (mV)")
   
    plt.show()


#firing rate as a function of input

def Exact_Integrate_and_Fire(I, t):
    """
    Args:
        I  : Input Current
        t : time
    Returns:
        Spike : Spike Count
        Spike_time : Spike time
        V_exact : Exact membrane potential
    """
    Spike = 0
    tau_m = 10
    R = 10
    t_isi = 0
    V_reset = E_L = -75
    V_exact = V_reset * np.ones(len(t))
    V_th = -50
    Spike_time = []

    for i in range(0, len(t)):
        V_exact[i] = E_L + R * I + (V_reset - E_L - R * I) * np.exp(-(t[i] - t_isi) / tau_m)

        # Threshold Reset
        if V_exact[i] > V_th:
            #setting the previous voltage to 0 and resetting the voltage to the reset potential 
            #previous voltage is set to 0 because it is the last voltage before the spike
            V_exact[i - 1] = 0
            V_exact[i] = V_reset
            t_isi = t[i]
            Spike = Spike + 1
            Spike_time = np.append(Spike_time, t[i])

    return Spike, Spike_time, V_exact


I_range = np.arange(2.0, 4.0, 0.1)
Spike_rate = np.ones(len(I_range))

for i, I in enumerate(I_range):
  Spike_rate[i], _, _ = Exact_Integrate_and_Fire(I, t)

with plt.xkcd():
  fig = plt.figure(figsize=(6, 4))
  plt.plot(I_range,Spike_rate)
  plt.xlabel('Input Current (nA)')
  plt.ylabel('Spikes per Second (Hz)')
  plt.show()
