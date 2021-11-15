import numpy as np
import matplotlib.pyplot as plt
import numba as nb

# Gathering data using the numpy method - using the 'same' procedure as for libc.so.6
@nb.njit
def gen_primary_rand():
    n = 300000000
    data = np.random.randint(0, 2**31-1, (n, 3))
    return data

def gen_numpy_rand():
    data = gen_primary_rand()
    data = data[np.max(data, axis=1) < 1e8]
    print(data)    
    np.savetxt("rand_points3.txt", data, delimiter=" ", fmt='%d')

    
#gen_numpy_rand() # Only call once to generate numpy rand data

data_title = ["supplied data", "libc.so.6", "numpy randint", "test"]
for i in range(1, 5):
    # Loading the random data points
    data_pnts = np.loadtxt("rand_points"+str(i)+".txt", delimiter=" ")
    print(np.shape(data_pnts))
    
    # The parameters for our 2d view
    a=1
    b=-0.5

    # Plotting a 2d view of our random numbers
    plt.scatter(a*data_pnts.T[0] + b*data_pnts.T[1], data_pnts.T[2], s=0.51)
    plt.title("Planes in the PRNG, data source: "+data_title[i-1])
    plt.xlabel("x-0.5y")
    plt.ylabel("z")
    plt.show()
    #plt.savefig("PlanesInPRNG"+str(i))
    plt.cla()
    plt.clf()
    plt.close()
    
# My libc.so.6 (Ubuntu) is indistinguishable from numpy's rand


