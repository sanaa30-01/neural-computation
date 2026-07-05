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
#random synaptic input current leads to random time course for membrane potential

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


#coding exercise 8: storing simulations in lists

#plotting multiple simulations (N=50) by storing in a list the voltage of each neuron at time t

np.random.seed(2020)

step_end = int(t_max/dt)

#initializing the number of neurons
n = 50

#like the previous exercises, we are intializing rest (V(0)) voltage as the leak potential term of the neuron
#in this case, we are using a list comprehension to initialize the voltage of ALL 50 neurons to the leak potential term
v_n = [el] * n

#using the xkcd style for the plot because it looks like a hand-drawn graph 
with plt.xkcd():
    plt.figure()
    plt.title("Membrane Potential of 50 neurons (random input)")
    plt.xlabel("Time (s)")
    plt.ylabel("Membrane Potential (V)")

    for step in range(step_end):

        t = step * dt

        #using a loop to iterate over each neuron and calculate the voltage of each neuron at time t
        #iterates over indices of v_n list
        for j in range(0, n):

            random_num = 2 * np.random.random() - 1
            i = i_mean * (1 + 0.1 * (t_max/dt) ** 0.5 * random_num)

            #saving the new voltage of the neuron in the list, for that particular neuron --> jth neuron by indexing j
            #basically same formula as previous exercises but with iterations using the list comprehension
            v_n[j] = v_n[j] + dt/tau * (el - v_n[j] + r * i)

        #plotting the voltage of all neurons at time t
        #[t] * n is a list of n elements all equal to t, all the x-values are the same time t but the y-values are the voltages of all neurons at that time t
        #this allows us to plot all the neurons at the same time t on the same vertical line --> shows different realizations of the random voltages of all neurons at that time t
        #the random element in this is the current input, but the number of neurons changes
        plt.plot([t] * n, v_n, color = 'k', alpha = 0.1, marker = '.')

    plt.show()


#coding exercise 9: plotting sample mean

np.random.seed(2020)

step_end = int(t_max/dt)
n = 50

v_n = [el] * n

plt.figure()
plt.title("Random Membrane Potential with Sample Mean")
plt.xlabel("Time (s)")
plt.ylabel("Membrane Potential (V)")

for step in range(step_end):
    t = step * dt
    
    for j in range(0, n):

        random_num = 2 * np.random.random() - 1
        i = i_mean * (1 + 0.1 * (t_max/dt) ** 0.5 * random_num)
        v_n[j] = v_n[j] + dt/tau * (el - v_n[j] + r * i)

    #calculating the sample mean of the voltages of all neurons at time t
    #basically the average of the voltages of all neurons for each time step t
    v_mean = sum(v_n) / n

    plt.plot([t] * n, v_n, color = 'k', alpha = 0.1, marker = '.')

    #plotting the sample mean of the voltages of all neurons at time t
    plt.plot(t, v_mean, color = 'C0', marker = '.')

plt.show()

#coding exercise 10: plotting sample standard deviation 

np.random.seed(2020)

step_end = int(t_max/dt)
n = 50

v_n = [el] * n

plt.figure()
plt.title(f"Membrane Potential with Sample Mean and Standard Deviation")
plt.xlabel("Time (s)")
plt.ylabel("Membrane Potential (V)")

for step in range(step_end):

    t = step * dt

    for j in range(0, n):

        random_num = 2 * np.random.random() - 1
        i = i_mean * (1 + 0.1 * (t_max/dt) ** 0.5 * random_num)

        v_n[j] = v_n[j] + dt/tau * (el - v_n[j] + r * i)

    v_mean = sum(v_n) / n

    #calculating the sample variance of the voltages of all neurons at time t
    #basically the variance of the voltages of all neurons for each time step t
    v_var_n = [(v - v_mean) ** 2 for v in v_n]    #used a list comprehension 
    v_var = sum(v_var_n) / (n-1)

    #calculating the sample standard deviation of the voltages of all neurons at time t
    #basically the standard deviation of the voltages of all neurons for each time step t
    v_std = np.sqrt(v_var)

    plt.plot([t] * n, v_n, color = 'k', alpha = 0.1, marker = '.')

    plt.plot(t, v_mean, color = 'C0', alpha = 0.8, marker = '.', markersize = 10)

    #plotting the sample standard deviation of the voltages of all neurons at time t --> upper and lower bounds 
    plt.plot(t, v_mean + v_std, color = 'C7', alpha = 0.8, marker = '.')
    plt.plot(t, v_mean - v_std, color = 'C7', alpha = 0.8, marker = '.')

plt.show()

#Coding exercise 11: rewriting with numpy

np.random.seed(2020)

#creating a time range from 0 to t_max with step size dt
step_end = int(t_max/dt) - 1

#creating an array of time points from 0 to t_max with step_end size and t_max as the endpoint (not including t_max)
#linspace is a function that creates an array of evenly spaced numbers over a specified interval
t_range = np.linspace(0, t_max, num=step_end, endpoint=False)

#ones is a function that creates an array of all ones of a specified size
#mutliplied by el to create an array of the neuron's resting voltage (el) for all time points
v = el * np.ones(step_end)

#creating an array of random numbers between -1 and 1 for all time points instead of just one random number for each time point
random_num = 2 * np.random.random(size=step_end) - 1
#creating an array of input currents for all time points
i = i_mean * (1 + 0.1 * (t_max/dt) ** 0.5 * random_num)

#using a loop to iterate over each time point
for step in range(step_end):

    #using indexing to access the previous time point's voltage and present current to calculate the new voltage
    #storing it in the array v at the current time point step
    v[step] = v[step - 1] + dt/tau * (el - v[step - 1] + r * i[step])

with plt.xkcd():
    plt.figure()
    plt.title("Membrane Potential with Numpy")
    plt.xlabel("Time (s)")
    plt.ylabel(f"Membrane Potential (V)")

    plt.plot(t_range, v, color = 'k', marker = '.')

    plt.show()

#Coding exercise 12: using enumerate and indexing
#aim: use enumerting for the synaptic current input --> get the index of the current and the value

np.random.seed(2020)

step_end = int(t_max/dt) - 1
t_range = np.linspace(0, t_max, num=step_end, endpoint=False)
v = el * np.ones(step_end)

random_num = 2 * np.random.random(size=step_end) - 1
i = i_mean * (1 + 0.1 * (t_max/dt) ** 0.5 * random_num)

for step, i_step in enumerate(i):

    if step == 0:
        continue

    v[step] = v[step - 1] + dt/tau * (el - v[step - 1] + r * i_step)

with plt. xkcd():
    plt.figure()
    plt.title("Membrane Potential with Enumeration")
    plt.xlabel("Time (s)")
    plt.ylabel("Membrane Potential (V)")

    plt.plot(t_range, v, color = 'k')

    plt.show()


#Coding exercise 13: using 2D arrays

np.random.seed(2020)

step_end = int(t_max/dt)
n = 50
t_range = np.linspace(0, t_max, num=step_end)

#creating a 2D array of the neuron's resting voltage for all neurons at all time points
#shape of the array is n neurons and step_end time points 
v_n = el * np.ones([n, step_end])

#creating a 2D array of random numbers for all neurons at all time points instead of just one random number for each neuron at each time point
random_num = 2 * np.random.random(size=[n, step_end]) - 1
i = i_mean * (1 + 0.1 * (t_max/dt) ** 0.5 * random_num)

#using a loop to iterate over each time point
#starting from 1 because we need to access the previous time point's voltage to calculate the new voltage
for step in range(1, step_end):

    v_n[:, step] = v_n[:, step - 1] + dt/tau * (el - v_n[:, step - 1] + r * i[:, step])

with plt.xkcd():
    plt.figure()
    plt.title("Membrane Potential of 50 neurons with 2D arrays")
    plt.xlabel("Time (s)")
    plt.ylabel("Membrane Potential (V)")
    
    #plotting the voltage of all neurons at all time points
    #using 2D arrays to plot the trajectories of all neurons as time progresses
    #T is the transpose of the array, so each column is one neuron's full time series and that allows for the trajectory of each neuron to be plotted 
    #transpose is required because the array is in the shape of n neurons and step_end time points, so we need to transpose it to plot all neurons at the same time point on the same vertical line
    plt.plot(t_range, v_n.T, color = 'k', alpha = 0.3)

    plt.show()


#coding exercise 14: plotting sample mean and standard deviation using 2D arrays

np.random.seed(2020)

step_end = int(t_max/dt)
n = 50
t_range = np.linspace(0, t_max, num=step_end)
v_n = el * np.ones([n, step_end])

random_num = 2 * np.random.random(size=[n, step_end]) - 1
i = i_mean * (1 + 0.1 * (t_max/dt) ** 0.5 * random_num)

for step in range(1, step_end):

    v_n[:, step] = v_n[:, step - 1] + dt/tau * (el - v_n[:, step - 1] + r * i[:, step])

#calculating the sample mean of the voltages of all neurons at all time points
#axs = 0 is used because we want to calculate the average of the voltages of all neurons at each time point (time points are the columns)
v_mean = np.mean(v_n, axis = 0)

#calculating the sample standard deviation of the voltages of all neurons at all time points
#axs = 0 is used because we want to calculate the standard deviation of the voltages of all neurons at each time point (time points are the columns)
v_std = np.std(v_n, axis = 0)

with plt.xkcd():
    plt.figure()
    plt.title("Membrane Potential with Sample Mean and Standard Deviation using 2D arrays")
    plt.xlabel("Time (s)")
    plt.ylabel("Membrane Potential (V)")

    plt.plot(t_range, v_n.T, color = 'k', alpha = 0.3)

    plt.plot(t_range, v_n[-1], color = 'k', alpha = 0.3, label = 'V(t)')
    plt.plot(t_range, v_mean, color = 'C0', alpha = 0.8, label = 'mean')
    plt.plot(t_range, v_mean + v_std, color = 'C7', alpha = 0.8)
    plt.plot(t_range, v_mean - v_std, color = 'C7', alpha = 0.8)

    plt.show()
