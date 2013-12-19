#Solution to Probit exercise
#This project: JG, JEH, YW
#This code: JG
#This draft: 19/12/2013

#Import packages
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize 
#Set seed
np.random.seed(0)

#Exercise 2.5
#Generate data
N = 5; T = 3; betak = .5; betan = .2; sigmaeps = 1.0; pi = .2; gamma = .8
sigmaeta = .2; covepseta = .3

kappa = []; z = []; y = []; eps = []; eta = []

for t in range(0,T):
    #kappa
    kappa.extend([np.random.uniform(0,5) for i in range(0,N)])
    #z
    z.extend([np.random.uniform(0,5) for i in range(0,N)])
    #y
    y.extend([np.random.uniform(0,10) for i in range(0,N)])
    
for i in range(0,N): 
    #eps,et
    epset = np.random.multivariate_normal([0,0],[[sigmaeps**2, covepseta],[sigmaeps**2, covepseta]])
    eps.append(epset[0])
    eta.append(epset[1])

eps = T*eps
eta = T*eta








    
    
    



    
    
    








