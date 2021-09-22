import numpy as np
import matplotlib.pyplot as plt

# A global variable which counts recursions
recursion_count = 0

# A global variable which counts number of calls to the integrated function
function_call_count = 0

def simpson(func, a, b, f_a, f_interm, f_b):
    """
    Obtains the three point integral simpson's estimate
    
    :param a: The integral start point
    :param b: The integral endpoint
    :param f_a: The function value at the start point
    :param f_interm: The function value at the intermediate  point
    :param f_b: The function value at the endpoint
    
    :return: Simpson's three point integral estimate
    """
    #Calculates the simpson integral
    simpson_integral = ((b-a)/6.0)*(f_a + 4.0*f_interm + f_b)
    return simpson_integral


def integrate_adaptive(func, a, b, tol, extra=None):  
    """
    Obtains the three point integral simpson's estimate
    
    :param a: The integral start point
    :param b: The integral endpoint
    :param tol: The error tolerance
    :param extra: Data from the previous run to avoid 
                  recalculating values
           extra[0:-2] is f_a, f_(a+b)/2 and f_b
           extra[-1] is the previous integral estimate

    :return: Simpson's three point integral estimate
    """
    # Access the recursion_counter as a global variable
    global recursion_count

    # Increment at each function call
    recursion_count += 1

    # Check if this is the first function call
    if np.atleast_1d(extra).any() == None:    
        # Calculate the required function values
        y = [0]*5
        for i in range(0, 5):
            y[i] = func(a + i*(b-a)/(len(y)-1))

        # Obtains a first estimate for the integral
        init_integral = simpson(func, a, b, y[0], y[2], y[4])

    # It is not the first function call
    else:
        # Calculate the required function values
        # Recover previously calculated values 
        y = [0]*5
        y[0] = extra[0]
        y[1] = func(a + 1*(b-a)/(len(y)-1))
        y[2] = extra[1]
        y[3] = func(a + 3*(b-a)/(len(y)-1))
        y[4] = extra[2]

        # Recover previously called integral as initial estimate
        init_integral = extra[-1]

    # Calculates a finer integral by splitting the region in two
    # Pass one useful function values to avoid re-calculating
    finer_integral_L = simpson(func, a, (a+b)/2.0, y[0], y[1], y[2])
    finer_integral_R = simpson(func, (a+b)/2.0, b, y[2], y[3], y[4])

    # Combine Left and Right region integrals
    finer_integral = finer_integral_L + finer_integral_R

    # Calculates the error based on how much the finer intregral
    # improved on the initial integral estimate
    err = np.abs(init_integral - finer_integral)

    # If the error is within tolerance return finer integral value
    if err <= tol:
        return finer_integral

    # If the error is too large feed back the left and right
    # regions to the function (recursive algo.)
    else:
        # Setup the extra data to avoid recalculating values
        extra_L = np.concatenate((y[0:3], [finer_integral_L]))
        extra_R = np.concatenate((y[2:], [finer_integral_R]))

        # Call the self-function by splitting the region in two
        recursion_integral_L = integrate_adaptive(func, a,
                                                    (a+b)/2.0, tol, extra_L)           
        recursion_integral_R = integrate_adaptive(func, (a+b)/2.0,
                                                  b, tol, extra_R)

        # return the sum of the left and right region integrals
        return recursion_integral_L + recursion_integral_R
        

def test_func(x):
    # This is verbose debugging - prints x used.
    print("X:", x)

    # Accesses the function call count as a global variable
    global function_call_count

    # Increment at each function call
    function_call_count += 1

    # return the function value at x
    return np.exp(x)


# Integrate the test function
integ = integrate_adaptive(test_func, -10, 10, 1.0e-7)

print("Recursion count:", recursion_count)
print("Function call count:", function_call_count)
print(integ)

# This saves 3 points for every recursion call
# As instead of calculating 0 -- 1 -- 2 -- 3 -- 4
# It only has to calculate 1 and 3
