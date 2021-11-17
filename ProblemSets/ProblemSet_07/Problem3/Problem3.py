import numpy as np
import matplotlib.pyplot as plt

def exponential_dist_PDF(x, alpha):
    return np.exp(-alpha*x)

# Generating uniform u,v.
n = 10000000
u = np.random.rand(n)
v = np.random.rand(n)*0.736

# Filtering out according to the ratios of uniform method
bound = np.sqrt(exponential_dist_PDF(v/u, 1.0))
accepted = u < bound
r = v/u

# Forming the exponential dist.
exp_dist = r[accepted]

# Plotting the histogram with an analytical line overlap
x = np.linspace(0, 10, 1000)
fig, ax = plt.subplots(1, figsize=(16, 9))
plt.plot(x, exponential_dist_PDF(x, 1), label="Analytical PDF")
plt.hist(exp_dist[np.abs(exp_dist) < 10], 1000, density=True, label="Histogram of the points generated")
plt.legend()
plt.tight_layout()
plt.savefig("Histogram")

# Plotting the accepted values to show our upper bound is correct
fig, ax = plt.subplots(1, figsize=(16, 9))
plt.scatter(u[accepted], v[accepted], s=0.1, label="Upper bound for v")
plt.plot([0, 1], [0.736, 0.736], color="red", linewidth=0.5, label="Accepted points")
plt.xlabel("u")
plt.ylabel("v")
plt.legend()
plt.tight_layout()
plt.savefig("UpperBound")

# Zoomed version of the above
fig, ax = plt.subplots(1, figsize=(16, 9))
plt.scatter(u[accepted], v[accepted], s=1, label="Upper bound for v")
plt.plot([0, 1], [0.736, 0.736], color="red", linewidth=0.5, label="Accepted points")
plt.ylim(0.734, 0.737)
plt.xlabel("u")
plt.ylabel("v")
plt.legend()
plt.tight_layout()
plt.savefig("Zoomed_UpperBound")

print("The proportion of accepted values", np.mean(accepted))
