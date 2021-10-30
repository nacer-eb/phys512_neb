import numpy as np
import matplotlib.pyplot as plt
import sys

# Setup imports
sys.path.insert(1, "../Problem1")
sys.path.insert(1, "../Problem2")

from Problem1 import apply_shift
from Problem2 import apply_correlation

# Obtain a sample gaussian
x = np.arange(-5, 5, 0.1)
gaussian = np.exp(-0.5*x**2)

# Shift the gaussian linearly and plot a correlation at each step
for i in range(0, 5):
    gaussian_shifted = apply_shift(gaussian, i*15)
    corr = apply_correlation(gaussian, gaussian_shifted)
    plt.plot(corr, color="black", alpha=i/6.0)
plt.savefig("correlation_shift")
