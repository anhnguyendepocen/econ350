''' Estimate the Static Model
    This project: JG, JEH, YW
    This code: YW
    This draft: 20/12/2013
'''

# Import packages
import numpy as np
import scipy.stats
import os
import json
import sys
from scipy.optimize import fmin_bfgs

# Set working directory
#os.chdir('/Users/ywang/Documents/TA/ECON350_Yike/StructuralEstimation/Handout2/Solution/Code/static')

# Import data
data = np.genfromtxt('data', dtype = 'float')
Y = data[:,0:3]
Z = data[:,3:6]
Kappa = data[:,6:9]
nn = data[:,9:12]
D = data[:,12:15]
W = data[:,15:18]

# Initial values for MLE
startVal = np.array([0.5, np.log(0.4), 0.4, 0.8, np.log(1.0), 0.3])

#beta_kappa = .5; sigma_eps = .4; beta_n + pi_ = .4; 
#gamma1 = .8; sigma_eta = 1; sigma_eps_eta = .3; 

'''Objective function (negative sample log-likelihood function with
   unconstrained parameters as inputs)
'''

def objFun(param,Y,Z,Kappa,nn,D,W):
    
    '''transfer unconstrained parameters into constrained parameters'''
    beta_kappa = param[0]
    sigma_eps = np.exp(param[1])
    beta_n_pi = param[2]
    gamma1 = param[3]
    sigma_eta = np.exp(param[4])
    sigma_eps_eta = param[5]
    
    '''calculate negative sample log-likelihood function'''
    
    '''calculate the latent function (decision rule)'''
    lat = Z*gamma1 - nn*(beta_n_pi) - Kappa*beta_kappa
 
    '''calculate the likelihood if D = 1'''
    sigma_xi_eta = sigma_eta**2 - sigma_eps_eta
    sigma_xi = np.sqrt(sigma_eta**2 + sigma_eps**2 - 2*sigma_eps_eta)
    temp1 = lat + (sigma_xi_eta/(sigma_eta**2))*(W-gamma1*Z)
    temp2 = np.sqrt(sigma_xi**2 - (sigma_xi_eta**2)/(sigma_eta**2))
    L1 = (1/sigma_eta)*scipy.stats.norm(0, 1).pdf((W-gamma1*Z)/sigma_eta)*scipy.stats.norm(0,1).cdf(temp1/temp2)

    '''calculate the likelihood if D = 0'''
    L0 = scipy.stats.norm(0,1).cdf(-lat/sigma_xi)

    '''calculate the negative sample log likelihood'''
    LL = D*np.log(L1) + (1-D)*np.log(L0)
    nLL = -sum(LL)

    '''finishing'''
    return nLL
    
''' Maximum Likelihood Estimation
'''
sys.stdout = open('Logging.txt', 'a')
rslt = fmin_bfgs(objFun, startVal, args = (Y,Z,Kappa,nn,D,W))
sys.stdout = sys.__stdout__

'''Construct dictionary with estimated parameter values
'''
mle = {}
mle['beta_kappa'] = rslt[0]
mle['sigma_eps'] = np.exp(rslt[1])
mle['beta_n_pi'] = rslt[2]
mle['gamma1'] = rslt[3]
mle['sigma_eta'] = np.exp(rslt[4])
mle['sigma_esp_eta'] = rslt[5]

with open('MLE_static.json','w') as file_:
    json.dump(mle,file_)
