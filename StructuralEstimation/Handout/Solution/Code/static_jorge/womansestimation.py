#Solution to Static Model: Estimation
#This project: JG, JEH, YW
#This code: JG
#This draft: 21/12/2013

#Import packages
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize 
#Set seed
np.random.seed(0)

#Time periods
T = 3

#Import data
data = np.genfromtxt('womansdata', dtype = 'float')
y = data[:,0:T]
z = data[:,T:2*T]
kappa = data[:,2*T:3*T]
n = data[:,3*T:4*T]
d = data[:,4*T:5*T]
w = data[:,5*T:6*T]

#Maximization
#Initial Condition
theta0 = np.array([0.5, np.log(0.4), 0.4, 0.8, np.log(1.0), 0.3])

#Define the negative of the likelihood function (the routine minimizes)
def logllk(theta):
    
    #Estimands
    betak = theta[0]
    sigmaeps = np.exp(theta[1])
    pibetan = theta[2]
    gamma = theta[3]
    sigmaeta = np.exp(theta[4])
    covepseta = theta[5]    
    
    #Locals
    covxieta = sigmaeta**2 -covepseta
    sigmaxi  = np.sqrt(sigmaeta**2 +sigmaeps**2 -2*covepseta)
    xi_star =  z*gamma -n*(pibetan) -kappa*betak
    
    #l_0
    l_0 = norm(0,1).cdf(-xi_star/sigmaxi)
    
    #l_1   
    pdf_l1 = norm(0, 1).pdf((w -z*gamma)/sigmaeta)   
    arg1_cdf_l1 = xi_star + (covxieta/(sigmaeta**2))*(w -z*gamma)
    arg2_cdf_l1 = np.sqrt(sigmaxi**2 -(covxieta**2)/(sigmaeta**2))
    cdf_l1 = norm(0,1).cdf(arg1_cdf_l1/arg2_cdf_l1) 
    
    l_1 = (1/sigmaeta)*pdf_l1*cdf_l1 
    
    return -np.sum(d*np.log(l_1) +(1-d)*np.log(l_0))

#Optimize
opt = minimize(logllk, theta0, method='Nelder-Mead', options={'disp': True, 'maxiter': 50000})
thetaopt = opt.x
thetaopt[1] = np.exp(thetaopt[1])
thetaopt[4] = np.exp(thetaopt[4])

#Print optimization information
print (thetaopt)
print (opt.success)
print (opt.message)