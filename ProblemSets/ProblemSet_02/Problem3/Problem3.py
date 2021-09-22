import numpy as np
import numpy.polynomial.chebyshev as cheb
import matplotlib.pyplot as plt

e_base = np.log2(np.e)

def mylog2_fit(deg, tol):
    """
    Fits a chebyshev polynomial to log base 2, from 0.5 to 1

    :param deg: The degree of the polynomial
    :param tol: The error tolerance for the fit

    :return: The chebyshev coefficients
    """
    # Obtains data for the fit (this is by default)
    x_true = np.linspace(0.5, 1, 100)
    # Rescaled from -1 to 1
    x_true_rescaled = (x_true - 0.75)*4
    y_true = np.log2(x_true)

    # Obtains the coef and residuals
    coef, res = cheb.chebfit(x_true_rescaled, y_true, deg, full=True)

    # If the residuals/error is too large repeat the fit with a higher deg.
    if res[0][0] > tol:
        return mylog2_fit(deg+1, tol)
    
    # If the error is small enough return the chebyshev coefs.
    return coef


def mylog2(x, coef):
    """
    Returns the natural logarithm of x, using a chebyshev fit

    :param deg: The point for which to obtain 'ln(x)'
    :param tol: The pre-fitted chebyshev coefficients

    :return: The estimate of ln(x)
    """
    # Obtain the mantissa and the exponent (in powers of two)
    m, expo = np.frexp(x)

    # Rescale the mantissa for our chebyshev polynomial
    m = (m - 0.75)*4.0

    # Estimate the natural log
    nat_log = (expo + cheb.chebval(m, coef))/e_base

    return nat_log


#### Now testing the code #### 

# x linspace for which to calculate ln(x) - avoiding ln(0)
x_space = np.linspace(1e-7, 100, 10000)

# Calculate the chebyshev coefs.
coef = mylog2_fit(3, 1e-15)

# Calculate using my function the ln(x) value
y = mylog2(x_space, coef)

# Calculate the actual value to see how effective this code is
y_true = np.log(x_space)

# Plot the chebyshev ln(x) estimate
plt.plot(x_space, y)
plt.title("Chebyshev natural logarithm estimate")
plt.show()

# Plot the error in the chebyshev ln(x) estimate
plt.scatter(x_space, np.abs(y-y_true), s=1)
plt.title("The error in the Chebyshev estimate")
plt.show()

print("The maximum error is:", np.max(np.abs(y-y_true)))
