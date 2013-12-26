''' Simulate the Dynamic Model
    This project: JG, JEH, YW
    This code: YW
    This draft: 23/12/2013
'''

# Import packages
import numpy as np
from scipy.stats import norm

# Set seed
np.random.seed(0)

# Parameters
N = 1000; T = 3; beta_kappa = .5; beta_n = .2; sigma_eps = 0.4; pi_ = .2; 
gamma1 = .8; gamma2 = .9; delta = 0.85; sigma_eta = 1.0; sigma_eps_eta = .3; 
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
'''

def W1(y,z,h,n):
    return y + gamma1*z + gamma2*h - pi_*n
    
def W0(y,kappa,n):
    return y + beta_kappa*kappa + beta_n*n
    
def xi_starT(z,h_T,n,kappa):
    return W1(0.0,z,h_T,n) - W0(0.0,kappa,n)

sigma_xi = np.sqrt(sigma_eta**2 + sigma_eps**2 - 2.0*sigma_eps_eta)   
sigma_eta_xi = sigma_eta**2 - sigma_eps_eta
sigma_eps_xi = sigma_eps_eta - sigma_eps**2
    
def EmaxT(EyT,z,h,n,kappa):
    part1 = 1.0 - norm(0,1).cdf(-xi_starT(z,h,n,kappa)/sigma_xi)
    pdf_ = norm(0,1).pdf(-xi_starT(z,h,n,kappa)/sigma_xi)
    cdf_ = norm(0,1).cdf(-xi_starT(z,h,n,kappa)/sigma_xi)
    part2 = W1(EyT,z,h,n) + (sigma_eta_xi/sigma_xi)*(pdf_/(1-cdf_))
    part3 = W0(EyT,kappa,n) - (sigma_eps_xi/sigma_xi)*(pdf_/cdf_)
    return part1*part2 + (1.0 - part1)*part3

def xi_starTm1(z,h_Tm1,n,kappa):
    rslt = W1(0.0,z,h_Tm1,n) - W0(0.0,kappa,n) + delta*(EmaxT(0.0,z,h_Tm1+1.0,n,kappa) - EmaxT(0.0,z,h_Tm1,n,kappa))
    return rslt
    
def EmaxTm1(EyTm1,z,h,n,kappa,EyT):
    part1 = 1.0 - norm(0,1).cdf(-xi_starTm1(z,h,n,kappa)/sigma_xi)
    pdf_ = norm(0,1).pdf(-xi_starTm1(z,h,n,kappa)/sigma_xi)
    cdf_ = norm(0,1).cdf(-xi_starTm1(z,h,n,kappa)/sigma_xi)
    part2 = W1(EyTm1,z,h,n) + delta*EmaxT(EyT,z,h + 1.0,n,kappa) + (sigma_eta_xi/sigma_xi)*(pdf_/(1.0 - cdf_))
    part3 = W0(EyTm1,kappa,n) + delta*EmaxT(EyT,z,h,n,kappa) - (sigma_eps_xi/sigma_xi)*(pdf_/cdf_)
    return part1*part2 + (1.0 - part1)*part3

def xi_starTm2(z,h,n,kappa):
    rslt = W1(0,z,h,n) - W0(0,kappa,n) + delta*(EmaxTm1(0.0,z,h+1.0,n,kappa,0.0) - EmaxTm1(0.0,z,h,n,kappa,0.0))
    return rslt

# Decision matrix
D = np.zeros((N,T))
# Work experience matrix
hh = np.zeros((N,T))

# Calculate D at T = 1
lat1 = xi_starTm2(Z[:,0],hh[:,0],nn[:,0],Kappa[:,0]) # latent variable function
xi1 = eta[:,0] - eps[:,0]
D[:,0] = xi1 > -lat1
hh[:,1] = D[:,0]
# Calculate D at T = 2
lat2 = xi_starTm1(Z[:,1],hh[:,1],nn[:,1],Kappa[:,1])   
xi2 = eta[:,1] - eps[:,1]
D[:,1] = xi2 > -lat2
hh[:,2] = hh[:,1] + D[:,1]
# Calculate D at T = 3
lat3 = xi_starT(Z[:,2],hh[:,2],nn[:,2],Kappa[:,2])
xi3 = eta[:,2] - eps[:,2]
D[:,2] = xi3 > -lat3

'''Individuals' wages:
note that the wages for women who chose not to work are set to zero
'''
W = D*(gamma1*Z + gamma2*hh + eta)
   
'''Export sample to *.txt file for further processing. 
'''
np.savetxt('data', np.column_stack((Y, Z, Kappa, nn, D, W, hh)), fmt= '%8.3f')





