#regression using least squares optimization
#fit simple linear models to data

#aim:
    #1. learn how to calculate the mean squared error (MSE)
    #2. explore how model parameters (slope) influence the MSE
    #3. learn how to find the optimal model parameter using least-squares optimization 

import numpy as np
import matplotlib.pyplot as plt

#mean squared error (MSE)

#generate simulated data

np.random.seed(121)

#setting parameters
theta = 1.2
n_sample = 30

#generating x values

#we know this is uniformly distributed between 0 and 10 because 10 * np.random.rand(n_sample) generates a random number between 0 and 10
x = 10 * np.random.rand(n_sample) 

#generating noise from a standard normal distribution with mean 0 and standard deviation 1
noise = np.random.randn(n_sample)

#generating y values --> this is the equation that we will eventually use to fit the model
y = theta * x + noise


#plotting the simulated data
fig, ax = plt.subplots()
ax.scatter(x, y)
ax.set(xlabel='x', ylabel='y');

plt.show()


#coding exercise 1: computing MSE

def mse(x, y, theta_hat):
    """Compute the mean squared error
    Args:
    x (ndarray): An array of shape (samples,) that contains the input values.
    y (ndarray): An array of shape (samples,) that contains the corresponding measurement values to the inputs.
    theta_hat (float): An estimate of the slope parameter

    Returns:
    float: The mean squared error of the data with the estimated parameter.
    """

    #computing the estimated y values
    y_hat = theta_hat * x

    #computing the mean squared error
    mse = np.mean((y - y_hat)**2)

    return mse

#computing the MSE for different theta_hat values
theta_hats = [0.75, 1.0, 1.5]
for theta_hat in theta_hats: 
    print(f"theta_hat of {theta_hat} has an MSE of {mse(x, y, theta_hat):.2f}")



#plotting the MSE for different theta_hat values on the scatter plot of the original data 
fig, axes = plt.subplots(ncols=3, figsize=(15, 5))
for theta_hat, ax in zip(theta_hats, axes):

  # True data
  ax.scatter(x, y, label='Observed')  # our data scatter plot

  # Compute and plot predictions
  y_hat = theta_hat * x
  ax.plot(x, y_hat, color='r', label='Fit')  # our estimated model

  ax.set(
      title= fr'$\hat{{\theta}}$= {theta_hat}, MSE = {np.mean((y - y_hat)**2):.2f}',
      xlabel='x',
      ylabel='y'
  );

axes[0].legend()
plt.show()




#coding exercise 2: solving for the optimal estimator 

def solve_optimal_estimator(x, y):
    """ Solve the normal equations to find the value of theta_hat that minimizes the MSE

    Args:
    x (ndarray): An array of shape (samples,) that contains the input values.
    y (ndarray): An array of shape (samples,) that contains the corresponding data values to the inputs. 

    Returns:
    float: the optimal value of theta_hat achieved after minimizing the MSE
    """

    theta_hat = (np.dot(x, y)) / (np.dot(x, x))

    return theta_hat


theta_hat = solve_optimal_estimator(x, y)
y_hat = theta_hat * x 

with plt.xkcd():
    fig, ax = plt.subplots()
    ax.scatter(x, y, label='Observed')
    ax.plot(x, y_hat, color='r', label='Fit')

    # plot residuals
    ymin = np.minimum(y, y_hat)  #lower tip of the vertical lines that represent the residuals, whichever is lower between the observed y values and the estimated y values
    ymax = np.maximum(y, y_hat)  #upper tip of the vertical lines that represent the residuals, whichever is higher between the observed y values and the estimated y values
    ax.vlines(x, ymin, ymax, 'g', alpha=0.5, label='Residuals')
    ax.set(
        title=fr"$\hat{{\theta}}$ = {theta_hat:0.2f}, MSE = {np.mean((y - y_hat)**2):.2f}",
        xlabel='x',
        ylabel='y'
    )
    
    ax.legend()
    plt.show() 