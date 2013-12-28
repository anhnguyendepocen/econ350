#Solution to Static Model: Simulation
#This project: JG, JEH, YW
#This code: JG
#This draft: 27/12/2013

#Import packages
import numpy as np
from scipy.stats import norm

#Set seed
np.random.seed(0)

#Model Parameters
N = 1000; T = 3; betak = .5; betan = .2; sigmaeps = 0.4; pi = .2; 
gamma1 = .8; sigmaeta = 1; covepseta = .3; gamma2 = .9; delta = .85;
covepsxi = covepseta - sigmaeps**2
covetaxi = sigmaeta**2 - covepseta
sigmaxi = np.sqrt(sigmaeta**2 + sigmaeps**2 - 2*covepseta)


#Simimulation Parameters
y_lb = 0.0; y_up = 10.0;
k_lb = 0.0; k_up = 5.0;
z_lb = 0.0; z_up = 5.0;
n_lb = 0; n_up = 3

#Simulate observed variables
#y
y = np.random.uniform(low=y_lb,high=y_up,size=(N,T))

#z,k,n
z = np.zeros((N,T))
kappa = np.zeros((N,T))
n = np.zeros((N,T))

z_N = np.random.uniform(low=z_lb,high=z_up,size=N)
kappa_N = np.random.uniform(low=k_lb,high=k_up,size=N)
n_N = np.random.random_integers(low=n_lb,high=n_up,size=N)

for t in range(T):
    z[:,t] = z_N
    kappa[:,t] = kappa_N
    n[:,t] = n_N

#Simulate unobserved variables
epseta = np.random.multivariate_normal([0,0],[[sigmaeps**2, covepseta],[covepseta, sigmaeta**2]],(N,T))
eps = epseta[:,:,0]
eta = epseta[:,:,1]


#Construct latent and observed decision
#Non-stochastic components of the utility function
def W1(z,h,n):
    return gamma1*z + gamma2*h - pi*n
    
def W0(kappa,n):
    return betak*kappa + betan*n

#xi_star @T
def xi_starTm0(z,h,n,kappa):
    return W1(z,h,n) - W0(kappa,n)

#Emax @T
def EmaxTm0(z,h,n,kappa):
    cdf1 = norm(0,1).cdf(xi_starTm0(z,h,n,kappa)/sigmaxi)
    pdf  = norm(0,1).pdf(xi_starTm0(z,h,n,kappa)/sigmaxi)
    cdf0 = norm(0,1).cdf(-xi_starTm0(z,h,n,kappa)/sigmaxi)
    return W1(z,kappa,n)*cdf1 + W0(kappa,n)*cdf0
    + sigmaxi+pdf
    
#xi_star @t = 1,...,T-1
for t in range(1,T):
    exec("def xi_starTm" + str(t) + "(z,h,n,kappa):\
    return W1(z,h,n) - W0(kappa,n)   \
    + delta*EmaxTm" + str(t-1) + "(z,h+1.0,n,kappa)\
    - delta*EmaxTm" + str(t-1) + "(z,h,n,kappa)")

#Emax @ t = T-1 (you can loop it similarly as xi_star before).
def EmaxTm1(z,h,n,kappa):
    cdf1 = norm(0,1).cdf(xi_starTm0(z,h,n,kappa)/sigmaxi)
    pdf  = norm(0,1).pdf(xi_starTm0(z,h,n,kappa)/sigmaxi)    
    cdf0 = norm(0,1).cdf(-xi_starTm0(z,h,n,kappa)/sigmaxi)    
    return (W1(z,kappa,n) + delta*EmaxTm0(z,h+1.0,n,kappa))*cdf1 
    + (W0(kappa,n) + delta*EmaxTm0(z,h+1.0,kappa))*cdf0
    + sigmaxi+pdf

xi_star = np.zeros((N,T)); d = np.zeros((N,T)); h = np.zeros((N,T))
xi = np.zeros((N,T))

for t in range(0,T):
    exec("xi_star[:," + str(t) + "] = xi_starTm" + str(T-1-t) 
    + "(z[:," + str(t) + "], h[:," + str(t) + "]\
    , n[:," + str(t) + "], kappa[:," + str(t) + "])")
    d[:,t] = eta[:,t] - eps[:,t] > xi_star[:,t]    
    if t < T-1:   
        h[:,t+1] = h[:,t] + d[:,t]
#wage
w = d*(gamma1*z + gamma2*h + eta)

np.savetxt('womanddata', np.column_stack((y, z, kappa, n, d, w, h)), fmt= '%8.3f')