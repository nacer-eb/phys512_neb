import numpy as np
import matplotlib.pyplot as plt


def apply_shift(f_x, m):
    """
    Applies a shift to f_x
    
    :param f_x: Array containing f(x) values
    :param m: The shift size

    :return: The shifted array f_x
    """
    # Obtain the length of the input array
    N = len(f_x)

    # Setup the kronecker delta
    delta_m = np.zeros(N)
    delta_m[m] = 1

    # Setup and array to store the shifted f_x
    # Convolution with a kronecker delta leads to a shift.
    f_x_shifted = np.zeros(N)
    for x in range(0, N):
        for chi in range(0, N):
            f_x_shifted[x] += delta_m[chi]*f_x[x-chi]

    return f_x_shifted


def main():
    """
    Output the answer to problem 1
    
    :return: None
    """

    # Obtain a gaussian
    x = np.arange(-5, 5, 0.1)
    gaussian = np.exp(-0.5*(x)**2)

    # Shift the gaussian
    gaussian_shifted = apply_shift(gaussian, 50)

    # Plot the shifted gaussian along with the inital gaussian
    plt.plot(gaussian, color="blue", label="initial")
    plt.plot(gaussian_shifted, color="orange", label="shifted")
    plt.legend()
    plt.savefig("gauss_shift")


# Avoid running when importing
if __name__ == "__main__":
    main()
