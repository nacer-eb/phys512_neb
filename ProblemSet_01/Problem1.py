import numpy as np
import matplotlib.pyplot as plt
import struct

# Check system architecture
is64bit = (struct.calcsize('P') == 8)
if is64bit:
    print("Python is running in 64bit architecture")
if not is64bit:
    print("Python is running in 32bit architecture")

############ Derivative function definitions ############
def calcDerivative(f, x, dx):
    """
    Calculates derivative using the 'symmetric' method
    
    :param f: the function to calculate the derivative of
    :param x: the x value at which the derivative must be 
              calculated.
    :param dx: the dx of the approximation.

    :return: The derivative value.
    """
    assert dx > 0 # Makes sure dx>0 so that the derivative does not blow up.

    df = f(x + dx) - f(x-dx)
    return df/(2.0*dx)


def combineDerivative(f, x, delta):
    """ 
    Combines derivatives to cancel the 3rd degree error
    
    :param f: the function to calculate the derivative of
    :param x: the x value at which the derivative must be 
              calculated.
    :param delta: the delta of the approximation.

    :return: The more accurate derivative value.
    """ 
    f1 = calcDerivative(f, x, delta)
    f2 = calcDerivative(f, x, 2.0*delta)

    accurate_derivative = (4*f1 - f2)/(3)
    return accurate_derivative


############ Sample functions definitions ############

def f_exp1(x):
    return np.exp(x)

def f_exp2(x):
    return np.exp(0.01*x)


############ Putting it all together ############

# I define by default the smallest fractional difference for a 64/32 bit system
if is64bit:
    eps = 1/np.power(2, 53)
if not is64bit:
    eps = 1/np.power(2, 24)

def getDerivativeFracErr(f, a, x, eps, verbose, delta_custom=None):
    if delta_custom is None: 
        delta = np.power(eps*eps*1.0/np.power(a, 10), 1.0/12.0)
    else:
        delta = delta_custom
        
    deriv = combineDerivative(f, x, delta)
    err = np.divide(deriv - a*f(x), a*f(x))
    if verbose == 2:
        plt.plot(x, deriv)
        plt.title("The derivative")
        plt.show()
    if verbose >= 1:
        plt.plot(x, err)
        plt.title("The fractional error in the derivative")
        plt.show()
        return delta, np.std(err)
    
    return np.std(err)

x1 = np.linspace(-1, 1, 1000)
chosen_delta1, std_err1 = getDerivativeFracErr(f_exp1, 1, x1, eps, 2)
print(std_err1)

x2 = np.linspace(-100, 100, 1000)
chosen_delta2, std_err2 = getDerivativeFracErr(f_exp2, 0.01, x2, eps, 2)
print(std_err2)


all_delta = np.linspace(chosen_delta1*0.001, chosen_delta1*4, 100)
all_err = np.zeros(100)
for i in range(0, len(all_delta)):
    all_err[i] = getDerivativeFracErr(f_exp1, 1, x1, eps, 0, all_delta[i])
plt.plot(all_delta, all_err)
plt.scatter(chosen_delta1, std_err1, color="red")
plt.show()


all_delta = np.linspace(chosen_delta2*0.01, chosen_delta2*4, 100)
all_err = np.zeros(100)
for i in range(0, len(all_delta)):
    all_err[i] = getDerivativeFracErr(f_exp2, 0.01, x2, eps, 0, all_delta[i])
plt.plot(all_delta, all_err)
plt.scatter(chosen_delta2, std_err2, color="red")
plt.show()

"""
x1 = np.linspace(-1, 1, 1000)

accurate_deriv_1 = combineDerivative(f_exp1, x1, delta1)

plt.plot(x1, accurate_deriv_1, color="orange")
plt.plot(x1, f_exp1(x1), color="blue")
plt.show()

plt.plot(x1,accurate_deriv_1 - f_exp1(x1), color="blue")
plt.show()

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

"""

