''' Estimate the Dynamic Model
    This project: JG, JEH, YW
    This code: YW
    This draft: 23/12/2013
'''

# Import packages
import numpy as np
import sys
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt 
import json

# Import data
data = np.genfromtxt('data', dtype = 'float')
Y = data[:,0:3]
Z = data[:,3:6]
Kappa = data[:,6:9]
nn = data[:,9:12]
D = data[:,12:15]
W = data[:,15:18]
hh = data[:,18:21]

'''Objective function (negative sample log-likelihood function with
   unconstrained parameters as inputs)
'''

def W1(y,z,h,n,param):
    gamma1 = param['gamma1']
    gamma2 = param['gamma2']
    pi_ = param['pi_']
    return y + gamma1*z + gamma2*h - pi_*n
    
def W0(y,kappa,n,param):
    beta_kappa = param['beta_kappa']
    beta_n = param['beta_n']
    return y + beta_kappa*kappa + beta_n*n
    
def xi_starT(z,h_T,n,kappa,param):
    return W1(0.0,z,h_T,n,param) - W0(0.0,kappa,n,param)
   
def EmaxT(EyT,z,h,n,kappa,param):
    sigma_xi = param['sigma_xi']
    sigma_eta_xi = param['sigma_eta_xi']
    sigma_eps_xi = param['sigma_eps_xi']
    part1 = 1.0 - norm(0,1).cdf(-xi_starT(z,h,n,kappa,param)/sigma_xi)
    pdf_ = norm(0,1).pdf(-xi_starT(z,h,n,kappa,param)/sigma_xi)
    cdf_ = norm(0,1).cdf(-xi_starT(z,h,n,kappa,param)/sigma_xi)
    part2 = W1(EyT,z,h,n,param) + (sigma_eta_xi/sigma_xi)*(pdf_/(1-cdf_))
    part3 = W0(EyT,kappa,n,param) - (sigma_eps_xi/sigma_xi)*(pdf_/cdf_)
    return part1*part2 + (1.0 - part1)*part3

def xi_starTm1(z,h_Tm1,n,kappa,param):
    rslt = W1(0.0,z,h_Tm1,n,param) - W0(0.0,kappa,n,param) + \
           param['delta']*(EmaxT(0.0,z,h_Tm1+1.0,n,kappa,param) - EmaxT(0.0,z,h_Tm1,n,kappa,param))
    return rslt
    
def EmaxTm1(EyTm1,z,h,n,kappa,EyT,param):
    sigma_xi = param['sigma_xi']
    sigma_eta_xi = param['sigma_eta_xi']
    sigma_eps_xi = param['sigma_eps_xi']
    delta = param['delta']
    part1 = 1.0 - norm(0,1).cdf(-xi_starTm1(z,h,n,kappa,param)/sigma_xi)
    pdf_ = norm(0,1).pdf(-xi_starTm1(z,h,n,kappa,param)/sigma_xi)
    cdf_ = norm(0,1).cdf(-xi_starTm1(z,h,n,kappa,param)/sigma_xi)
    part2 = W1(EyTm1,z,h,n,param) + delta*EmaxT(EyT,z,h + 1.0,n,kappa,param) + (sigma_eta_xi/sigma_xi)*(pdf_/(1.0 - cdf_))
    part3 = W0(EyTm1,kappa,n,param) + delta*EmaxT(EyT,z,h,n,kappa,param) - (sigma_eps_xi/sigma_xi)*(pdf_/cdf_)
    return part1*part2 + (1.0 - part1)*part3

def xi_starTm2(z,h,n,kappa,param):
    delta = param['delta']
    rslt = W1(0.0,z,h,n,param) - W0(0.0,kappa,n,param) + delta*(EmaxTm1(0.0,z,h+1.0,n,kappa,0.0,param) - EmaxTm1(0.0,z,h,n,kappa,0.0,param))
    return rslt

def objFun(par):
    '''transform unconstrained parameters into constrained parameters'''
    param = {}
    param['beta_kappa'] = par[0]
    param['sigma_eps'] = np.exp(par[1])
    param['beta_n'] = par[2]
    param['gamma1'] = par[3]
    param['gamma2'] = par[4]
    param['delta'] = par[5]
    param['sigma_eta'] = np.exp(par[6])
    param['sigma_eps_eta'] = (np.exp(par[7])/(1.0+np.exp(par[7])) - 0.5)*2.0*np.exp(par[1])*np.exp(par[6])
    param['pi_'] = par[8]
    
    '''calculate the variance-covariance matrix'''
    param['sigma_xi'] = np.sqrt(param['sigma_eta']**2 + param['sigma_eps']**2 - 2.0*param['sigma_eps_eta'])   
    param['sigma_eta_xi'] = param['sigma_eta']**2 - param['sigma_eps_eta']
    param['sigma_eps_xi'] = param['sigma_eps_eta'] - param['sigma_eps']**2
    
    '''calculate negative sample log-likelihood function'''   

    '''calculate the latent function (decision rule)'''
    lat1 = xi_starTm2(Z[:,0],hh[:,0],nn[:,0],Kappa[:,0],param) # T = 1
    lat2 = xi_starTm1(Z[:,1],hh[:,1],nn[:,1],Kappa[:,1],param) # T = 2
    lat3 = xi_starT(Z[:,2],hh[:,2],nn[:,2],Kappa[:,2],param) # T = 3
    lat = np.column_stack((lat1, lat2, lat3))
    
    '''calculate the likelihood if D = 1'''
    temp1 = lat + (param['sigma_eta_xi']/param['sigma_eta']**2)*(W-Z*param['gamma1']-hh*param['gamma2'])
    temp2 = np.sqrt(param['sigma_xi']**2 - (param['sigma_eta_xi']**2)/(param['sigma_eta']**2))
    L1 = (1.0/param['sigma_eta'])*norm(0,1).pdf((W-param['gamma1']*Z-param['gamma2']*hh)/param['sigma_eta'])*norm(0,1).cdf(temp1/temp2)
    
    '''calculate the likelihood if D = 0'''
    L0 = norm(0,1).cdf(-lat/param['sigma_xi'])
    
    '''calculate the negative sample log likelihood'''
    LL = D*np.log(L1) + (1.0-D)*np.log(L0)
    nLL = -sum(LL)
    
    '''finishing'''
    return nLL
    
''' Maximum Likelihood Estimation
'''

# Initial values for MLE
startVal = np.array([0.5, np.log(0.4), 0.2, 0.8, 0.9, 0.85, np.log(1.0), np.log((0.3+0.4*1.0)/(0.4*1.0-0.3)),0.2])
#startVal = np.array([0.2, np.log(0.6), 0.4, 0.5, 0.7, 0.75, np.log(0.8), np.log((0.3+0.4*1.0)/(0.4*1.0-0.3)),0.3])

'''
param_for_plot = startVal
grid_for_plot = np.linspace(0.0,0.4,100)
fvalue_for_plot = np.zeros(100)
for i in range(100):
    param_for_plot[8] = grid_for_plot[i]
    fvalue_for_plot[i] = objFun(param_for_plot)
plt.plot(grid_for_plot,fvalue_for_plot)
'''

sys.stdout = open('Logging.txt', 'a')
rslt = minimize(objFun, startVal, method='nelder-mead', options={'disp': True})
sys.stdout = sys.__stdout__


'''Construct dictionary with estimated parameter values
'''
mle = {}
mle['beta_kappa'] = rslt.x[0]
mle['sigma_eps'] = np.exp(rslt.x[1])
#mle['beta_n'] = rslt.x[2]
mle['gamma1'] = rslt.x[3]
mle['gamma2'] = rslt.x[4]
mle['delta'] = rslt.x[5]
mle['sigma_eta'] = np.exp(rslt.x[6])
mle['sigma_esp_eta'] = (np.exp(rslt.x[7])/(1.0+np.exp(rslt.x[7])) - 0.5)*2.0*np.exp(rslt.x[1])*np.exp(rslt.x[6])
#mle['pi_'] = rslt.x[8]
mle['beta_n_pi'] = rslt.x[2] + rslt.x[8]

with open('MLE_dynamic.json','w') as file_:
    json.dump(mle,file_)

    