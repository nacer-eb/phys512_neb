import numpy as np
import matplotlib.pyplot as plt

def conv_safe(f, g):
    """
    Compute the convolution between f, g without wrapping around
    
    :param f: Array containing f(x) values
    :param g: Array containing g(x) values

    :return: The safe convolution of f and g 
    """
    # Obtain the length of each f, g array
    N_f = len(f)
    N_g = len(g)

    # Make sure len (f) >= len (g) fill with zeros when needed 
    if N_f < N_g:
        f_new = np.zeros(N_g)
        f_new[0:N_f] = f
        f = f_new

    # Update the length of f stored
    N_f = len(f)

    # Fill g with zeros at the end
    # The total array length is N_f + N_g
    g_new = np.zeros(N_f + N_g)
    g_new[0:N_g] = g
    g = g_new

    # Do the convolution as usual
    h = np.zeros(N_f+N_g)
    for x in range(0, N_f+N_g):
        for chi in range(0, N_g):
            h[x] += f[chi]*g[x-chi]


    # Return the result of the convolution (len: N_f + N_g)
    return h

# Obtain two gaussians with the same starting points
x_1 = np.arange(-5, 5, 0.1)
gauss_1 = np.exp(-0.5*(x_1)**2)

x_2 = np.arange(-5, 5, 0.1)
gauss_2 = np.exp(-0.5*(x_2+3)**2)

# Apply a 'safe' convolution
conv = conv_safe(gauss_1, gauss_2)

# Plot the result
plt.plot(conv)
plt.title("Safe convolution")
plt.savefig("safe_conv")

