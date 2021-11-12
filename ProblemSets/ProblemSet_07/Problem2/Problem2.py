import numpy as np
import matplotlib.pyplot as plt

n=10000000

large_sample = np.random.random(n)*10-5

def normed_cauchy_PDF(x, gamma):
    return (1.0/(gamma*np.pi))*(gamma**2/(x**2 + gamma**2))


def inv_cauchy_CDF(q, gamma):
    return gamma*np.tan(np.pi*(q-0.5))


def generate_cauchy(gamma):
    unif_rands = np.random.rand(n)
    return inv_cauchy_CDF(unif_rands, gamma)


def exponential_dist_PDF(x, alpha):
    return np.exp(-alpha*x)





cauchy_gen = generate_cauchy(1.0)
cauchy_gen_filtered = cauchy_gen[np.abs(cauchy_gen) < 1000]

cauchy_gen_final = cauchy_gen_filtered[cauchy_gen_filtered > 0]

acceptance_rate = (1.0/np.pi)*exponential_dist_PDF(cauchy_gen_final, 1)/normed_cauchy_PDF(cauchy_gen_final, 1)
acceptance_toss = np.random.rand(len(cauchy_gen_final)) < acceptance_rate

exponential_dist = cauchy_gen_final[acceptance_toss]

print(np.mean(acceptance_toss))

"""

x = np.linspace(-50, 50, 10000)
plt.plot(x, normed_cauchy_PDF(x, 1))
plt.hist(cauchy_gen_filtered, bins=5000, density=True)
plt.xlim(-5, 5)
plt.show()
"""

x = np.linspace(0, 100, 10000)
plt.plot(x, np.pi*normed_cauchy_PDF(x, 1))
plt.plot(x, exponential_dist_PDF(x, 1), color="orange")
plt.show()



x = np.linspace(0, 10, 10000)
plt.plot(x, exponential_dist_PDF(x, 1))
plt.hist(exponential_dist, bins=1000, density=True)
plt.xlim(0, 10)
plt.show()



