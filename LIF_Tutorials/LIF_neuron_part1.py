#LIF = leaky integrate and fire neuron model --> also connected to point neuron model approaches 
#aim = visualize evolution of membrane potential in time and extract statistical properties
#work with membrane potential (voltage = v(t)), leak term, and spikes

import numpy as np
import matplotlib.pyplot as plt


#coding exercise 1: defining parameters for the LIF model

t_max = 150e-3  #s
dt = 1e-3  #s
tau = 20e-3  #s
el = -60e-3  #mV
vr = -70e-3  #mV
vth = -50e-3  #mV
r = 100e6  #ohm
i_mean = 25e-11  #A

print(t_max, dt, tau, el, vr, vth, r, i_mean)


#coding exercise 2: simulating an input current

for step in range(10):
    
    #computing t for the equation 
    t = step * dt

    #current equation
    i = i_mean * (1 + np.sin(2 * np.pi * t/0.01))

    #printing the time calculated upto 3 decimals and the current calculated upto 4 decimals but in exponential form
    print(f"t = {t:.3f} s, i = {i:.4e} A")


#coding exercise 4: simulating membrane potential

step_end = 10
v = el

for step in range(step_end):

    #compute value of t
    t = step * dt
    i = i_mean * (1 + np.sin(2 * np.pi * t/0.01))

    #calculating the new value of v using the discrete time integration equation
    v = v + (dt/tau) * (el - v + (r * i))

    print(f"t = {t:.3f} s, i = {v:.4e} mV")


#coding exercise 5: plotting the current

step_end = 25

#initializing the figure we want to plot the current on
plt.figure()
plt.title("Current")
plt.xlabel("Time (s)")
plt.ylabel("Current (A)")

for step in range(step_end):

    t = step * dt
    i = i_mean * (1 + np.sin(2 * np.pi * t/0.01))

    #plotting the current
    plt.plot(t, i, color = 'k', marker = 'o')  #or could be written as plt.plot(t, i, 'ko')

#display the plot
plt.show() 

#coding exercise 6: plotting the membrane potential 

#had to calculate step end because we are using a discrete time step and we need to know how many steps we need to take to reach the maximum time
#the maximum time here is t_max and the time step is delta t which is 0.001 s which is dt in the code 
step_end = int(t_max/dt)

v = el

plt.figure()
plt.title("Membrane Potential")
plt.xlabel("Time (s)")
plt.ylabel("Membrane Potential (V)")

for step in range(step_end):

    t = step * dt
    i = i_mean * (1 + np.sin(2 * np.pi * t/0.01))
    v = v + dt/tau * (el - v + (r * i))

    plt.plot(t, v, color = 'k', marker = 'o')

plt.show()


#coding exercise 7: adding randomness to current to plot membrane potential graph

np.random.seed(2020)

step_end = int(t_max/dt)
v = el

plt.figure()
plt.title("Vm with random I(t)")
plt.xlabel("Time (s)")
plt.ylabel("Membrane Potential (V)")

for step in range(step_end):

    t = step * dt
    
    #generating a random number between -1 and 1 
    #since np.random.random() generates a random number between 0 and 1 (uses the uniform distribution), we need to multiply by 2 and subtract by 1 to get max and min values of -1 and 1.
    random_num = 2 * np.random.random() - 1

    i = i_mean * (1 + 0.1 * ((t_max/dt)**0.5) * random_num)

    v = v + dt/tau * (el - v + r * i)

    plt.plot(t, v, color = 'k', marker = '.')

plt.show()