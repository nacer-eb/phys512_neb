import numpy as np
import matplotlib.pyplot as plt


def apply_correlation(f, g):
    """
    Compute the correlation between f, g
    
    :param f: Array containing f(x) values
    :param g: Array containing g(x) values

    :return: The correlation of f and g 
    """
    # Compute the fft of f and g
    f_fft, g_fft = np.fft.fft(f), np.fft.fft(g)

    # Compute the conjugate of g_fft
    conj_g_fft = np.conjugate(g_fft)

    # Compute the correlation using the fourier space shortcut
    f_g_correlation = np.real(np.fft.ifft(f_fft*conj_g_fft))

    # Return the correlation
    return f_g_correlation


# Output answer for problem 2
def main():
    # Setup the trial gaussian function
    x = np.arange(-5, 5, 0.1)
    f = np.exp(-0.5*x**2)

    # Apply the self-correlation
    corr = apply_correlation(f, f)

    # Plot the correlation
    plt.plot(f, label="f")
    plt.plot(corr, label="correlation")
    plt.legend()
    plt.savefig("correlation_gaussian_self")

    
# Does not run when importing
if __name__ == "__main__":
    main()
