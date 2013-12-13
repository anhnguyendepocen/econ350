#Warm-up exercise solver
#Section 1
#This project: JG, JEH, YW
#This code: JG
#This draft: 12/12/2013

#Import packages
import matplotlib.pyplot as mplot
import os
import random
#Set seed
random.seed(0)

#Change to my principal path
os.chdir(os.environ["jorge"])
os.chdir("econ350/StructuralEstimation/Warmup/Solution/Data")


#1.1 Basic continous function through a grid
n = 1000
x = linspace(0, 2*pi, n+1)
s = sin(x)

mplot.plot(x,s)
mplot.title("$\sin(x)$")
mplot.xlabel('x'); 
mplot.savefig("sin.png")
mplot.close()

#1.4 Trapezoidal integration
def trapezoidalint(f, a, b, n):
    h = (b-a)/float(n)
    I = f(a) + f(b)
    for k in xrange(1, n, 1):
        x = a + k*h
        I += 2*f(x)
    I *= h/2
    return I

#1.5 Trapezoidal integration of the sine function
# Calculate the closed form integral
cfi_sin = -cos(pi) + cos(0)
print cfi_sin
#Calculate the approximations
for n in 10, 50, 100, 500:
    approx_trapint_sin = trapezoidalint(sin, 0, pi, n)
    print approx_trapint_sin

#1.6 Monte Carlo Integration
def mcint(f, a, b, n):
    s = 0
    for i in range(n):
        x = random.uniform(a, b)
        s += f(x)
    I = (float(b-a)/n)*s
    return I

#1.7 
for n in 2, 50, 100, 500, 1000:
    approx_mcint_sin = mcint(sin, 0, pi, n)
    print approx_mcint_sin