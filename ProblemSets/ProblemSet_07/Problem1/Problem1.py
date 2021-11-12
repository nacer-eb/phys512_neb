import numpy as np
import matplotlib.pyplot as plt

data_pnts = np.loadtxt("rand_points.txt", delimiter=" ")[0:100000]

print(np.shape(data_pnts))

"""
fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.scatter(data_pnts.T[0], data_pnts.T[1], data_pnts.T[2], s=1)
plt.show()
"""


a=1
b=-0.5

plt.scatter(a*data_pnts.T[0] + b*data_pnts.T[1], data_pnts.T[2], s=1)
plt.title("Planes in the PRNG")
plt.xlabel("x-0.5y")
plt.ylabel("z")
plt.show()
#plt.savefig("PlanesInPRNG")
plt.cla()
plt.clf()
plt.close()

# Clean this up
# Could not reproduce this with my libc
