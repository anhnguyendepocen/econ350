
#Warm-up exercise solver
#This project: JG, JEH, YW
#This code: JG
#This draft: 12/11/2013

#Import packages
import matplotlib.pyplot as mplot
import os

#Change to my principal path
os.chdir(os.environ["jorge"])
os.chdir("econ350/StructuralEstimation/Warmup/Solution/Output")


#1.1 Basic Function through a grid

n = 1000
x = linspace(0, 2*pi, n+1)
s = sin(x)

mplot.plot(x,s)
mplot.title("$\sin(x)$")
mplot.xlabel('x'); 
mplot.savefig("sin.png")
mplot.close()

