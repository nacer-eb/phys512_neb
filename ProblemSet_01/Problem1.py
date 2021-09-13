import numpy as np
import matplotlib.pyplot as plt
import platform

print("Python version is:", platform.architecture())

"""
Calculates derivative using the 'symmetric' method
    
:param f: the function to calculate the derivative of
:param x: the x value at which the derivative must be 
              calculated.
:param dx: the dx of the approximation.

:return: The derivative value.
"""
def calcDerivative(f, x, dx):
    assert dx > 0 # Makes sure dx>0 so that the derivative does not blow up.

    df = f(x + dx) - f(x-dx)
    return df/(2.0*dx)


""" 
Combines derivatives to cancel the 3rd degree error
    
:param f: the function to calculate the derivative of
:param x: the x value at which the derivative must be 
              calculated.
:param delta: the delta of the approximation.

:return: The more accurate derivative value.
""" 
def combineDerivative(f, x, delta):
    f1 = calcDerivative(f, x, delta)
    f2 = calcDerivative(f, x, 2.0*delta)

    accurate_derivative = (4*f1 - f2)/(3)
    return accurate_derivative

def f_exp1(x):
    return np.exp(x)

def f_exp2(x):
    return np.exp(0.01*x)

# I define by default the smallest fractional for a 64-bit system.
eps = 1e-16 # or 1e-7 for 32-bit systems
a1 = 1 # The parameter of exp(ax)
delta1 = np.power(180*eps*eps, 1.0/12.0)

"""
x1 = np.linspace(-1, 1, 1000)

accurate_deriv_1 = combineDerivative(f_exp1, x1, delta1)

plt.plot(x1, accurate_deriv_1, color="orange")
plt.plot(x1, f_exp1(x1), color="blue")
plt.show()

plt.plot(x1,accurate_deriv_1 - f_exp1(x1), color="blue")
plt.show()

"""
##

a2 = 0.01  # The parameter of exp(ax)
delta2 = np.power(180*eps*eps*1.0/np.power(a2, 10), 1.0/12.0)
print(delta2)
x2 = np.linspace(-100, 100, 1000)

accurate_deriv_2 = combineDerivative(f_exp2, x2, delta2)

plt.scatter(x2, accurate_deriv_2, color="orange")
plt.scatter(x2, a2*f_exp2(x2), color="blue")
plt.show()

plt.scatter(x2,(accurate_deriv_2 - a2*f_exp2(x2))/(a2*f_exp2(x2)), color="blue")
plt.show()


