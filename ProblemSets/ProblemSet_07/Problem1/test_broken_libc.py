import numpy as np
import ctypes
import numba as nb

# Setup the lib to use and the rand function
mylib=ctypes.cdll.LoadLibrary("libc.so.6")
rand=mylib.rand
rand.argtypes=[]
rand.restype=ctypes.c_int


# Generate a large number of random numbers using the above defined func.
@nb.njit
def get_rands_nb(vals):
    n=len(vals)
    for i in range(n):
        vals[i]=rand()
    return vals

def get_rands(n):
    vec=np.empty(n,dtype='int32')
    get_rands_nb(vec)
    return vec

# Get a list of rands
n = 300000000
vec=get_rands(n*3)

# Reshape the list into rows of (N, 3)
vv=np.reshape(vec,[n,3])

# Keep only entire rows which are under a threshold
maxval=1e8
vv2=vv[np.max(vv,axis=1)<maxval,:]

#Save the data
np.savetxt("rand_points4.txt", vv2, delimiter=" ", fmt='%d')

