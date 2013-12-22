''' Estimate the Static Model
    This project: JG, JEH, YW
    This code: YW
    This draft: 20/12/2013
'''

# Import packages
import numpy as np
import os
import json
import sys
from scipy.optimize import minimize 
from scipy.stats import norm

# Set working directory
#os.chdir('/Users/ywang/Documents/TA/ECON350_Yike/StructuralEstimation/Handout2/Solution/Code/static_yike1')

# Import data
data = np.genfromtxt('data', dtype = 'float')
Y = data[:,0:3]
Z = data[:,3:6]
Kappa = data[:,6:9]
nn = data[:,9:12]
D = data[:,12:15]
W = data[:,15:18]

# Initial values for MLE
#startVal = np.array([0.5, np.log(0.4), 0.4, 0.8, np.log(1.0),np.log((0.3+0.4*1.0)/(0.4*1.0-0.3))])
startVal = np.array([0.7, np.log(0.2), 0.5, 0.9, np.log(1.0),np.log((0.3+0.4*1.0)/(0.4*1.0-0.3))])

'''Objective function (negative sample log-likelihood function with
   unconstrained parameters as inputs)
'''

def objFun(param):
    '''transform unconstrained parameters into constrained parameters'''
    beta_kappa = param[0]
    sigma_eps = np.exp(param[1])
    beta_n_pi = param[2]
    gamma1 = param[3]
    sigma_eta = np.exp(param[4])
    sigma_eps_eta = (np.exp(param[5])/(1.0+np.exp(param[5])) - 0.5)*2.0*np.exp(param[1])*np.exp(param[4])
    '''calculate negative sample log-likelihood function'''
    
    '''calculate the latent function (decision rule)'''
    lat = Z*gamma1 - nn*(beta_n_pi) - Kappa*beta_kappa
     
    '''calculate the likelihood if D = 1'''
    sigma_xi_eta = sigma_eta**2 - sigma_eps_eta
    sigma_xi_2 = sigma_eta**2 + sigma_eps**2 - 2.0*sigma_eps_eta
    if not sigma_xi_2 > 0: 
        print sigma_xi_2
    assert (sigma_xi_2 > 0)
    sigma_xi = np.sqrt(sigma_xi_2)
    temp1 = lat + (sigma_xi_eta/(sigma_eta**2))*(W-gamma1*Z)
    temp2_2 = sigma_xi**2 - (sigma_xi_eta**2)/(sigma_eta**2)
    #assert (temp2_2 > 0)
    temp2 = np.sqrt(temp2_2)
    L1 = (1.0/sigma_eta)*norm(0,1).pdf((W-gamma1*Z)/sigma_eta)*norm(0,1).cdf(temp1/temp2)
    
    '''calculate the likelihood if D = 0'''
    L0 = norm(0,1).cdf(-lat/sigma_xi)
    
    '''calculate the negative sample log likelihood'''
    LL = D*np.log(L1) + (1.0-D)*np.log(L0)
    nLL = -sum(LL)
    
    '''finishing'''
    return nLL
    
''' Notice that an implicit bound for sigma_eps_eta is 
    (-simga_eps * sigma_eta, sigma_eps*sigma_eta)
    so, one way to transform an unconstrained parameter to satisfy
    this bound is the following: 
    (exp(x)/(1+exp(x)) - 1/2)*2*sigma_eps*sigma_eta
    '''
    
''' Maximum Likelihood Estimation
'''
sys.stdout = open('Logging.txt', 'a')
rslt = minimize(objFun, startVal, method='nelder-mead', options={'disp': True})
sys.stdout = sys.__stdout__

'''Construct dictionary with estimated parameter values
'''
mle = {}
mle['beta_kappa'] = rslt.x[0]
mle['sigma_eps'] = np.exp(rslt.x[1])
mle['beta_n_pi'] = rslt.x[2]
mle['gamma1'] = rslt.x[3]
mle['sigma_eta'] = np.exp(rslt.x[4])
mle['sigma_esp_eta'] = (np.exp(rslt.x[5])/(1+np.exp(rslt.x[5])) - 0.5)*2.0*np.exp(rslt.x[1])*np.exp(rslt.x[4])

with open('MLE_static.json','w') as file_:
    json.dump(mle,file_)
