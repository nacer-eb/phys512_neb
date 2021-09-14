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

############ Fractional error function definition ############

def getDerivativeFracErr(f, a, x, eps, verbose, delta_custom=None):
    """ 
    Calculates the derivative and obtains the std from the 'fractional err'
    as we expect the error to scale with f(x) in this case.
    
    :param f: the function to calculate the derivative of
    :param a: the exponential parameter of f
    :param x: the x value at which the derivative must be 
              calculated.
    :param verbise: verbose level == 2 - Plot the derivative approximation
                            level >= 1 - Plot the error in the derivative
                            level == 0 - Plot nothing
                            
    :param delta: the delta of the approximation, if None uses 
                  the calculated formula

    :return: The delta used (if verbose >= 1) 
             and the std error (for verbose > 0)
    """ 
    if delta_custom is None: 
        delta = np.power(eps*eps*225.0/np.power(a, 10), 1.0/10.0)
    else:
        delta = delta_custom
        
    deriv = combineDerivative(f, x, delta)
    err = np.divide(deriv - a*f(x),  a*f(x))
    if verbose == 2:
        plt.plot(x, deriv)
        plt.title("The derivative")
        plt.show()
    if verbose >= 1:
        plt.scatter(x, np.abs(err), s=1)
        plt.title("The fractional error in the derivative")
        plt.show()
        return delta, np.std(err)
    
    return np.std(err)


############ Putting it all together ############

# I define by default the smallest fractional difference for a 64/32 bit system
if is64bit:
    eps = 1/np.power(2, 53)
if not is64bit:
    eps = 1/np.power(2, 24)

  
# Calculates derivative for f(x), shows the error over the x range
# and shows that the delta used approximately minimizes the error
# this is shown by plotting the std_err over a range of possible delta.

#Here f(x) = exp(x)
x1 = np.linspace(-1, 1, 1000) # The x values for which df/dx is approximated

#Verbose calculation of the derivative, err, and std_err.
# (also returns the dela used)
chosen_delta1, std_err1 = getDerivativeFracErr(f_exp1, 1, x1, eps, 2) 

# Delta values for which we calculate std_err, and show our
# chosen_delta is close to the minima.
all_delta = np.linspace(0.0002, 5, 100)
all_err = np.zeros(100)
for i in range(0, len(all_delta)):
    all_err[i] = getDerivativeFracErr(f_exp1, 1, x1, eps, 0, all_delta[i])
plt.plot(all_delta, all_err)
plt.scatter(chosen_delta1, std_err1, color="red", s=8,
            label="The calculated delta")
plt.legend()
plt.ylabel("The std_err of the derivative")
plt.xlabel("Delta")
plt.show()


# Same as the above, but done for f(x) = exp(0.01*x)
x2 = np.linspace(-100, 100, 1000)
chosen_delta2, std_err2 = getDerivativeFracErr(f_exp2, 0.01, x2, eps, 2)

all_delta = np.linspace(0.02,500, 100)
all_err = np.zeros(100)
for i in range(0, len(all_delta)):
    all_err[i] = getDerivativeFracErr(f_exp2, 0.01, x2, eps, 0, all_delta[i])
plt.plot(all_delta, all_err)
plt.scatter(chosen_delta2, std_err2, color="red", s=8,
            label="The calculated delta")
plt.legend()
plt.ylabel("The std_err of the derivative")
plt.xlabel("Delta")
plt.show()


