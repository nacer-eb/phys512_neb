import numpy as np
import matplotlib.pyplot as plt

def rk4_step(func, x, y, h):
    k1 = h * func(x, y)
    k2 = h * func(x + h/2.0, y + k1/2.0)
    k3 = h * func(x + h/2.0, y + k2/2.0)
    k4 = h * func(x + h, y + k3)

    return y + (k1 + 2.0*k2 + 2.0*k3 + k4)/6.0


def rk4_stepd(func, x, y, h):
    y_h = rk4_step(func, x, y, h)
    y_h_half_stp_1 = rk4_step(func, x, y, h/2.0)
    y_h_half_stp_2 = rk4_step(func, x+h/2.0, y_h_half_stp_1, h/2.0)
    
    #Since the leading error is h⁵, we can write
    
    return (np.power(0.5, 5.0)*y_h - y_h_half_stp_2)/(np.power(0.5, 5.0) - 1.0)


def func(x, y):
    return y/(1.0+x*x)

h = 0.5*0.1
x = np.arange(-20, 20, h)
x_dtl = np.arange(-20, 20, 0.001)

y0 = 1
y = np.zeros(len(x)) + y0

for i in range(0, len(x)-1):
    y[i+1] = rk4_stepd(func, x[i], y[i], h)

plt.scatter(x, y, s=4, color="red")
plt.plot(x_dtl, np.exp(np.arctan(x_dtl))*np.exp(-np.arctan(-20)))
plt.show()

plt.scatter(x, np.abs(y-np.exp(np.arctan(x))*np.exp(-np.arctan(-20))), s=4, color="red")
plt.show()
