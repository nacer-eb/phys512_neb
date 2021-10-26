import numpy as np
import matplotlib.pyplot as plt

def conv_safe(f, g):
    N_f = len(f)
    N_g = len(g)

    if N_f < N_g:
        f_new = np.zeros(N_g)
        f_new[0:N_f] = f
        f = f_new
        
    N_f = len(f)

    g_new = np.zeros(N_f + N_g)
    g_new[0:N_g] = g
    g = g_new

    h = np.zeros(N_f+N_g)
    for x in range(0, N_f+N_g):
        for chi in range(0, N_g):
            h[x] += f[chi]*g[x-chi]
            
    return h[N_f:]

x_1 = np.arange(-5, 6, 0.1)
gauss_1 = np.exp(-0.5*x_1[:-10]**2)

x_2 = np.arange(-5, 5, 0.1)
gauss_2 = np.exp(-0.5*(x_2-0)**2)

#plt.plot(x_2, gauss_2)

conv = conv_safe(gauss_1, gauss_2)

plt.plot(conv)

conv = conv_safe(gauss_2, gauss_1)

plt.plot( conv)
plt.show()

