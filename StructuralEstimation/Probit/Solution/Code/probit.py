#Solution to Probit exercise
#This project: JG, JEH, YW
#This code: JG
#This draft: 13/12/2013

#Import packages
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize 
#Set seed
np.random.seed(0)

#Exercise 1.3
#Generate data
N = 100
X1 = np.array([np.random.lognormal(3,1.5) for i in range(0,N)])
X2 = np.array([np.random.randint(0,2) for i in range(0,N)])
X = zip(X1,X2)
E = np.array([np.random.normal(0,1) for i in range(0,N)])

#Set arbitrary beta and construct latent
beta = np.array([-.3,2])
Xbeta = np.dot(X,beta)
y_star = Xbeta + E

#Construct latent and observed decision
d = []
for i in range(0,N):
    if y_star[i] >= 0:
        d.append(1)
    else:
        d.append(0)
d = np.array(d)

#Define the negative of the likelihood function (the routine minimizes)
def llk(B):
    XB = np.dot(X,B)
    l_0 = log(norm.cdf(-XB))
    l_1 = log(norm.cdf(XB))
    for i in range(0,N):
        if l_0[i] == -inf:
            l_0[i] = 1e-30
        elif l_1[i] == -inf:
            l_1[i] = 1e-30
    return -sum(d*l_1 + (1-d)*l_0)

#Initial Condition
beta0 = np.array([-.3,2.0])
 
#Optimize
opt = minimize(llk, beta0, method='BFGS', options={'disp': True, 'maxiter': 50000})

#Print optimization information
print (opt.x)
print (opt.success)
print (opt.message)