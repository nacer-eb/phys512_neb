import numpy as np
import matplotlib.pyplot as plt

func_call_count = 0

def rk4_step(func, x, y, h):
    """
    Calculates a single rk4 integration steps
    
    :param func: The function to integrate
    :param x: The independant variable, this is often time t in some systems.
    :param y: The initial function value. 
              (Can be interpreted as the previous integration step result)

    :return: The integral of func at x.
    """
    # Calculate the k coefficients 
    k1 = h * func(x, y)
    k2 = h * func(x + h/2.0, y + k1/2.0)
    k3 = h * func(x + h/2.0, y + k2/2.0)
    k4 = h * func(x + h, y + k3)

    # Adds the current step to the previous y value
    # And returns the integral at x.
    return y + (k1 + 2.0*k2 + 2.0*k3 + k4)/6.0


def rk4_stepd(func, x, y, h):
    """
    Calculates a single rk4 integration steps, by comparing a full step size 
    to a half step and eliminating the leading-order error term.
    
    :param func: The function to integrate
    :param x: The independant variable, this is often time t in some systems.
    :param y: The function value at x. 
              (Can be interpreted as the previous integration step result)

    :return: The integral of func at x.
    """
    # Calculate the integral with a full step h.
    y_h = rk4_step(func, x, y, h)

    # Calculate the integral with two half steps h/2.0
    y_h_half_stp_1 = rk4_step(func, x, y, h/2.0)
    y_h_half_stp_2 = rk4_step(func, x+h/2.0, y_h_half_stp_1, h/2.0)
    
    # Since the (accumulated) leading error is h⁴
    # We cancel the h⁴ terms as such
    # Then return the integral at x.
    return (y_h - np.power(2, 4)*y_h_half_stp_2)/(1 - np.power(2, 4) )


def func_to_integ(x, y):
    """
    The function to integrate. (Supports arrays)
    
    :param x: The independant variable, this is often time t in some systems.
    :param y: The integral at x.
              
    :return: The value of the function to integrate at (x, y)
    """
    # Keep count of the number of function calls
    global func_call_count
    func_call_count += 1

    # Return the function value
    return y/(1.0+x*x)


def true_ans(x):
    """
    The true answer of the integral.
    
    :param x: The independant variable, this is often time t in some systems.
   
    :return: The value of the integral at (x, y)
    """
    return np.exp(np.arctan(x))*np.exp(-np.arctan(-20))

# First for the initial method.
# Set h and the x values on which to integrate. 
h = 0.05
x = np.arange(-20, 20, h)

# Repeat the setup for the second  method.
# Making sure to use h*3 so that the number of function calls are equal.
h_d = h*3
x_d = np.arange(np.min(x), np.max(x), h_d)

# Setting up the initial integral value (at x=-20)
# Setting up the y, y_d arrays, to hold integral values.
y0 = 1
y = np.zeros(len(x)) + y0
y_d = np.zeros(len(x_d)) + y0

# First compute the integral using the first method.
for i in range(0, len(x)-1):
    y[i+1] = rk4_step(func_to_integ, x[i], y[i], h)
    
print("The first method calls the function " , func_call_count, "times")

# Reset the function call count for method 2.
# Then compute the integral using the second method.
func_call_count = 0
for i in range(0, len(x_d)-1):
    y_d[i+1] = rk4_stepd(func_to_integ, x_d[i], y_d[i], h_d)
    
print("The second method calls the function ", func_call_count, "times")

# Plot the residuals
plt.scatter(x, np.abs(y-true_ans(x)), s=1, color="red", label="Method 1")
plt.scatter(x_d, np.abs(y_d-true_ans(x_d)), s=1, color="blue", label="Method 2")
plt.title("The residuals for both methods")
plt.ylabel("Absolute error")
plt.xlabel("x value")
plt.legend()
plt.savefig("Problem1_Residuals.png")

