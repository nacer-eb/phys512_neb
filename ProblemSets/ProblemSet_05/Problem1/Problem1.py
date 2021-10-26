import numpy as np
import matplotlib.pyplot as plt


def apply_shift(f_x, m, method = 0):

    N = len(f_x)
    sigma_m = np.zeros(N)
    sigma_m[m] = 1

    f_x_shifted = np.zeros(N)
    for x in range(0, N):
        for chi in range(0, N):
            f_x_shifted[x] += sigma_m[chi]*f_x[x-chi]

    return f_x_shifted


# Output the answer for problem 1
def main():
    x = np.arange(-5, 5, 0.1)
    gaussian = np.exp(-0.5*(x)**2)

    gaussian_shifted = apply_shift(gaussian, 50)
    
    plt.plot(gaussian, color="blue", label="initial")
    plt.plot(gaussian_shifted, color="orange", label="shifted")
    plt.legend()
    plt.savefig("gauss_shift")


# Avoid running when importing
if __name__ == "__main__":
    main()
