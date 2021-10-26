import numpy as np
import matplotlib.pyplot as plt


def apply_correlation(f, g):
    f_fft, g_fft = np.fft.fft(f), np.fft.fft(g)

    conj_g_fft = np.conjugate(g_fft)

    f_g_correlation = np.real(np.fft.ifft(f_fft*conj_g_fft))

    return f_g_correlation


# Output answer for problem 2
def main():
    x = np.arange(-5, 5, 0.1)
    f = np.exp(-0.5*x**2)
    
    corr = apply_correlation(f, f)
    
    plt.plot(f, label="f")
    plt.plot(corr, label="correlation")
    plt.legend()
    plt.savefig("correlation_gaussian_self")

    
# Does not run when importing
if __name__ == "__main__":
    main()
