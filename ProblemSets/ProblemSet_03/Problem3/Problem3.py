import numpy as np
import matplotlib.pyplot as plt

# First get the data
data = np.loadtxt("dish_zenith.txt", delimiter=" ").T

# Define the m coef-vector and A matrix
m = np.zeros(4) # a, beta, gamma, c
A = np.zeros((475, 4))

# Define the matrix using the linear combinations of
# a, beta, gamma, c that form z
A.T[0] = np.square(data[0]) + np.square(data[1])
A.T[1] = -2*data[0]
A.T[2] = -2*data[1]
A.T[3] = np.ones(475)

# Invert A^T * A - We assume uniform uncertainty on the data points
AT_A_inv = np.linalg.inv(np.matmul(A.T, A))

# Calculate the optimal parameters
m = np.matmul(np.matmul(AT_A_inv, A.T), data[2])

# Use our parameters to obtain a, x0, y0, z0
a = m[0]
x0 = m[1]/m[0]
y0 = m[2]/m[0]
z0 = m[3]-x0*x0*a-y0*y0*a

print("The parameters (a, x0, y0, z0) are: ", a, x0, y0, z0)

# One way to evaluate the parabola func.
# Here evaluated at the x,y points to compare model to data
z_eval = np.matmul(A, m)

# Another way to evaluate the parabola func.
def parabola_func(x, y, a, x0, y0, z0):
    return z0 + a*(np.square(x-x0) + np.square(y-y0))


# Sanity check - Plot that compares the data to the model
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

x = np.linspace(np.min(data[0]), np.max(data[0]), 50)
y = np.linspace(np.min(data[1]), np.max(data[1]), 50)

xx, yy = np.meshgrid(x,y)

xx = xx.ravel()
yy = yy.ravel()

zz = parabola_func(xx, yy, a, x0, y0, z0)

ax.scatter(xx, yy, zz, s=1, color="red", label="Model")
ax.scatter(data[0], data[1], data[2], s=1, color="blue", label="Data")
plt.legend()
plt.title("Comparison of the model to the data")
plt.savefig("Quick3D_Plot")
plt.cla()
plt.clf()
plt.close()


# Calculating the cov matrix
# Since we assume diagonal data uncertainty with uniform weights
data_var = np.std(data[2] - z_eval)**2
cov_mat = AT_A_inv*data_var

# Calculate the err in a
err_a = np.sqrt(cov_mat[0, 0])

# Calculate the err in x0 and y0
err_x0 = np.sqrt( np.square(x0)*(cov_mat[1, 1]/m[1]**2 + cov_mat[0, 0]/m[0]**2
                          - 2*cov_mat[0, 1]/(m[0]*m[1])) )

err_y0 = np.sqrt( np.square(y0)*(cov_mat[2, 2]/m[2]**2 + cov_mat[0, 0]/m[0]**2
                          - 2*cov_mat[0, 2]/(m[0]*m[2])) )

# Calculating the error in z0.
err_z0 = np.sqrt(cov_mat[3, 3]
                 + (x0**2 + y0**2)**2 * err_a**2
                 + (2*a*x0)**2 * err_x0**2
                 + (2*a*y0)**2 * err_y0**2)


print("Error in a", err_a)
print("Error in x_0", err_x0)
print("Error in y_0", err_y0)
print("Error in z_0", err_z0)

# Note for self, I verified all these results using scipy and they makes sense.
