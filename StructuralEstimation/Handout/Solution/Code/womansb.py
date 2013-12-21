#Solution Static Model: Simulation
#This project: JG, JEH, YW
#This code: JG
#This draft: 21/12/2013

#Import packages
import numpy as np

#Generate data
N = 1000; T = 3; betak = .5; betan = .2; sigmaeps = .4; pi = .2; gamma = .8
sigmaeta = 1; covepseta = .3
 
#kappa
kappa = [np.random.uniform(0,5) for i in range(0,N*T)]
#z
z = [np.random.uniform(0,5) for i in range(0,N*T)]
#y
y = [np.random.uniform(0,10) for i in range(0,N*T)]

#eps,eta    
eps = []; eta = []
for i in range(0,N*T): 
    epset = np.random.multivariate_normal([0,0],[[sigmaeps**2, covepseta],[covepseta, sigmaeta**2]])
    eps.append(epset[0])
    eta.append(epset[1])

#n
n = [np.random.randint(0,4) for i in range(0,N)]
n = T*n

#Construct the latent and the observed decision
w  = np.array(z)*gamma + np.array(eta)
v = np.array(z)*gamma -np.array(n)*(pi+betan) -np.array(kappa)*betak
+np.array(eta) -np.array(eps)
d = v >= 0

np.savetxt('data_jorge', np.column_stack((y, z, kappa, n, d, w)), fmt= '%8.3f')