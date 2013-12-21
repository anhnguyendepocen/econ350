''' Simulate the Static Model
    This project: JG, JEH, YW
    This code: YW
    This draft: 20/12/2013
'''

# Import packages
import numpy as np
import os

# Set working directory
os.chdir('/Users/ywang/Documents/TA/ECON350_Yike/StructuralEstimation/Handout2/Solution/Code/static_yike')

# Set seed
np.random.seed(0)

# Parameters
N = 1000; T = 3; beta_kappa = .5; beta_n = .2; sigma_eps = 0.4; pi_ = .2; 
gamma1 = .8; sigma_eta = 1; sigma_eps_eta = .3; 
y_lb = 0.0; y_up = 10.0;
k_lb = 0.0; k_up = 5.0;
z_lb = 0.0; z_up = 5.0;
n_lb = 0; n_up = 3

# Simulate observables
Y = np.random.uniform(low=y_lb,high=y_up,size=(N,T))
Z = np.zeros((N,T))
Kappa = np.zeros((N,T))
nn = np.zeros((N,T))
Z_temp = np.random.uniform(low=z_lb,high=z_up,size=N)
Kappa_temp = np.random.uniform(low=k_lb,high=k_up,size=N)
nn_temp = np.random.random_integers(low=n_lb,high=n_up,size=N)
for t in range(T):
    Z[:,t] = Z_temp
    Kappa[:,t] = Kappa_temp
    nn[:,t] = nn_temp

# Simulate unobservables
covs = np.diag([sigma_eta**2,sigma_eps**2])
covs[0,1] = 0.3
covs[1,0] = 0.3
means = [0.0, 0.0]
unobs = np.random.multivariate_normal(means,covs,(N,T))
eta = unobs[:,:,0]
eps = unobs[:,:,1]

'''Simulate individuals' decisions: 
first calculate U1, U0, and the latent variable function 
'''
U1 = Y + gamma1*Z + eta - pi_*nn
U0 = Y + beta_kappa*Kappa + beta_n*nn + eps
lat = U1 - U0
D = lat > 0

'''Simulate individuals' wages:
note that the wages for women who chose not to work are set to zero
'''
W = D*(gamma1*Z + eta)

'''Export sample to *.txt file for further processing. 
'''
np.savetxt('data', np.column_stack((Y, Z, Kappa, nn, D, W)), fmt= '%8.3f')





