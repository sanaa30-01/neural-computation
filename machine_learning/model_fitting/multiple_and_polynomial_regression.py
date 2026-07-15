#generalizing linear regression to multiple and polynomial regression

import numpy as np
import matplotlib.pyplot as plt


#using simulated data of d = 2 for multiple linear regression
#exploration dataset for theta = [0, -2, -3] and N = 40 noisy samples 

#simulating the data
# Set random seed for reproducibility
np.random.seed(1234)

# Set parameters
theta = [0, -2, -3]
n_samples = 40

# Draw x and calculate y

#number of regressors are the number of weights (theta values) we want to estimate 
n_regressors = len(theta)

#creating the design matrix X
#x0 is a column of ones to represent the intercept term
x0 = np.ones((n_samples, 1))
#x1 and x2 are random uniform distributions between -2 and 2 
x1 = np.random.uniform(-2, 2, (n_samples, 1)) 
x2 = np.random.uniform(-2, 2, (n_samples, 1))
X = np.hstack((x0, x1, x2))
noise = np.random.randn(n_samples)
y = X @ theta + noise


ax = plt.subplot(projection='3d')
ax.plot(X[:,1], X[:,2], y, '.')

ax.set(
    xlabel='$x_1$',
    ylabel='$x_2$',
    zlabel='y'
)
plt.tight_layout()
plt.show() 


#coding exercise 1: ordinary least squares estimator from the design matrix X and the target vector y

def ordinary_least_squares(X, y):
    """Ordinary least squares estimator for linear regression.
    
    Args:
       X (ndarray): design matrix of shape (n_samples, n_regressors)
       y (ndarray): vector of measurements of shape (n_samples)
    
    Returns:
       ndarray: estimated parameter values of shape (n_regressors)

    """

    theta_hat = np.linalg.inv(X.T @ X) @ X.T @ y

    return theta_hat

theta_hat = ordinary_least_squares(X, y)

print(theta_hat)


#computing y hat and plotting the results on a predicted plane

theta_hat = ordinary_least_squares(X, y)
y_hat = X @ theta_hat

#computing MSE
print(f"MSE = {np.mean((y - y_hat)**2):.2f}") 

#plotting the results on a predicted plane
theta_hat = ordinary_least_squares(X, y)
xx, yy = np.mgrid[-2:2:50j, -2:2:50j]
y_hat_grid = np.array([xx.flatten(), yy.flatten()]).T @ theta_hat[1:]
y_hat_grid = y_hat_grid.reshape((50, 50))

ax = plt.subplot(projection='3d')
ax.plot(X[:, 1], X[:, 2], y, '.')
ax.plot_surface(xx, yy, y_hat_grid, linewidth=0, alpha=0.5, color='C1',
                cmap=plt.get_cmap('coolwarm'))

for i in range(len(X)):
  ax.plot((X[i, 1], X[i, 1]),
          (X[i, 2], X[i, 2]),
          (y[i], y_hat[i]),
          'g-', alpha=.5)

ax.set(
    xlabel='$x_1$',
    ylabel='$x_2$',
    zlabel='y'
)
plt.tight_layout()
plt.show()
