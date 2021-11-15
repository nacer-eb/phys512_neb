import numpy as np
import matplotlib.pyplot as plt

# Generating Probability Density Functions
# The lorentzian PDF
def normed_cauchy_PDF(x, gamma):
    return (1.0/(gamma*np.pi))*(1.0/((x/gamma)**2 + 1))


# The exponential PDF
def exponential_dist_PDF(x, alpha):
    return alpha*np.exp(-alpha*x)


# The Lorentzian inverseCDF
def inv_cauchy_CDF(q, gamma):
    return gamma*np.tan(np.pi*(q-0.5))


# Generate the Cauchy distribution
def generate_cauchy(gamma, n):
    unif_rands = np.random.rand(n)
    return inv_cauchy_CDF(unif_rands, gamma)


# The parameters of the distributions
GAMMA = 0.808
ALPHA = 1.0

# Generate the cauchy dist. (using inverse CDF) and filter out the negative side
cauchy_gen = generate_cauchy(GAMMA, 10000000)
cauchy_gen_final = cauchy_gen[cauchy_gen > 0]

# Get the acceptance rate and filter out part of the cauchy dist. points
acceptance_rate = normed_cauchy_PDF(0, GAMMA)/exponential_dist_PDF(0, ALPHA) * exponential_dist_PDF(cauchy_gen_final, ALPHA) / normed_cauchy_PDF(cauchy_gen_final, GAMMA)
acceptance_mask = np.random.rand(len(cauchy_gen_final)) < acceptance_rate

# Use the above to then get the exponential dist.
exponential_dist = cauchy_gen_final[acceptance_mask]

# Verbose info
print("The proportion of points accepted", np.mean(acceptance_mask))
print("Number of points generated", len(exponential_dist))

# Plot to show the initial distribution is always above the target distribution
x = np.linspace(0, 20, 1000)
plt.plot(x, normed_cauchy_PDF(x, GAMMA)/normed_cauchy_PDF(0, GAMMA) - exponential_dist_PDF(x, ALPHA)/exponential_dist_PDF(0, ALPHA), color="orange")
plt.ylabel("Initial Distribution - Target Distribution")
plt.savefig("InitialVTargetDist_diff")
plt.cla()
plt.clf()
plt.close()

# Plot the histogram of the generated exponential distribution - compared to the analytical dist.
plt.hist(exponential_dist, bins=1000, density=True, label="Generated Distribution")
plt.plot(x, exponential_dist_PDF(x, ALPHA), label="Exponential Distribution - Analytical")
plt.ylabel("Normalized Distribution")
plt.legend()
plt.savefig("ExponentialDistGen_Hist")
plt.cla()
plt.clf()
plt.close()

