import numpy as np
import matplotlib.pyplot as plt
import struct

# Check system architecture
# and set the smallest fractional difference (epsilon) accordingly

is64bit = (struct.calcsize('P') == 8)
if is64bit:
    eps = 1.0/(np.power(2, 53))
    print("Python is running in 64bit architecture")
    
if not is64bit:
    eps = 1.0/(np.power(2, 23))
    print("Python is running in 32bit architecture")


############ Derivative function definitions (same as problem1) ############
def calcDerivative(f, x, dx):
    """
    Calculates derivative using the 'symmetric' method
    
    :param f: the function to calculate the derivative of
    :param x: the x value at which the derivative must be 
              calculated.
    :param dx: the dx of the approximation.

    :return: The derivative value.
    """
    assert dx.all() > 0 # Makes sure dx>0 so that the derivative does not blow up.

    df = f(x + dx) - f(x-dx)
    return df/(2.0*dx)


def calc3rdDerivative(f, x, dx):
    """
    Calculates 3rd derivative using the 'symmetric' method
    
    :param f: the function to calculate the derivative of
    :param x: the x value at which the derivative must be 
              calculated.
    :param dx: the dx of the approximation.

    :return: The 3rd derivative value.
    """
    d3f = f(x+3*dx)-3*(f(x+dx)-f(x-dx))-f(x-3*dx)
    return d3f/(8*dx*dx*dx)


def ndiff(fun, x, full=False):
    """
    Calculates derivative using the 'symmetric' method, by first 
    estimating the 3rd derivative
    
    :param fun: the function to calculate the derivative of
    :param x: the x value at which the derivative must be 
              calculated.
    :param full: The verbose level.

    :return: For full == True: returns the derivative, the calculated dx
                               and the estimated error.
             For full == False: returns only the derivative.
    """
    # We use this to estimate the 3rd derivative assuming f(x)/f'''(x) -> 1
    init_dx = np.power(np.sqrt(18)*eps, 1.0/4.0)
    deriv_3 = calc3rdDerivative(fun, x, init_dx)

    # We use this to approximate the actual derivative
    optimal_dx = np.power(np.abs(np.sqrt(18)*eps*(np.divide(fun(x), deriv_3))), 1.0/4.0)
    deriv_1 = calcDerivative(fun, x, optimal_dx)

    if full:
        # If full we also want to approximate the error;
        err = np.sqrt(np.square(np.multiply(deriv_3, np.square(optimal_dx))/6)
                      + np.square(np.divide(eps*fun(x), optimal_dx)))
        return deriv_1, optimal_dx, err
    
    return deriv_1


############ Test functions to test the above functions ############

def test_func_sin(x):
    return np.sin(x)

def test_func_actual_derivative(x):
    return np.cos(x)


############ Putting it all together with the above test functions ############

x = np.linspace(-14, 14, 200)
deriv, dx, err = ndiff(test_func_sin, x, True)

plt.plot(x, deriv, color="red", label="Derivative approximation")
plt.plot(x, test_func_actual_derivative(x), label="Actual derivative")
plt.legend()
plt.show()

plt.plot(x, err)
plt.title("The error of the derivative")
plt.show()
