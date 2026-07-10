import numpy as np
import matplotlib.pyplot as plt

#population equation 
#the rate of change of population is proportional to the population itself

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
t = np.arange(0, 10, 0.1)

with plt.xkcd():

    p = np.exp(0.3 * t)

    fig = plt.figure(figsize=(6, 4))
    plt.plot(t, p)
    plt.xlabel("Time (years)")
    plt.ylabel("Population (millions)")

    plt.show()