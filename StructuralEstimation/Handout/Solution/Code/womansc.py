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

# Import Yike's data
data = np.genfromtxt('data_yike', dtype = 'float')
y = data[:,0:3]
z = data[:,3:6]
kappa = data[:,6:9]
n = data[:,9:12]
d = data[:,12:15]
w = data[:,15:18]

#Import Jorge's data 
#data = np.genfromtxt('data_jorge', dtype = 'float')
#y = data[:,0]
#z = data[:,1]
#kappa = data[:,2]
#n = data[:,3]
#d = data[:,4]
#w = data[:,5]

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
opt = minimize(logllk, theta0, method='BFGS', options={'disp': True, 'maxiter': 50000})

#Print optimization information
print (opt.x)
print (opt.success)
print (opt.message)