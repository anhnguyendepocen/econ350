#This draft: 13/01/2014
#This code: Jorge

import numpy as np


arr2 = np.random.randn(10,5)
print type(arr2), arr2.shape, arr2.ndim
print arr2

sliceRow = arr2[2,:]
indexRow = arr2[2,[0,1,2,3,4]]
indexRow[0] = 99
print arr2[2,:]
sliceRow[0] = 99
print arr2[2,:]