import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.integrate

##### The custom integration code #####
#### TAKEN FROM MY PROBLEM 2 #####

# A global variable which counts recursions
recursion_count = 0

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
        
##### End of the custom integration code #####

# Define parameters initial values
z = 0
R = 1
q = 1e-11
rho = q/(4*np.pi)
epsilon_0 = 8.85e-12

# A function that calculates the actual spherical electric field at x-away from the center
def calc_actual_spherical_E(x):
    x = np.atleast_1d(x)
    E = [0]*len(x)
    for i in range(len(x)):
        if x[i] <= R:
            E[i] = 0
        if x[i] > R:
            E[i] = q/(4*np.pi*epsilon_0*x[i]*x[i])
    return E

# The function to integrate to obtain the spherical field (without constants)
def integrand_func(theta):
    numerator = np.sin(theta)*(z-R*np.cos(theta))
    denominator = np.power((R*R + z*z - 2*R*z*np.cos(theta)), 3.0/2.0)
    
    return numerator/denominator

# The rectified integrand; avoids 0/0 values.
def integrand_func_rectified(theta):
    numerator = np.sin(theta)*(z-R*np.cos(theta))
    if numerator == 0:
        return 0.0
    
    denominator = np.power((R*R + z*z - 2*R*z*np.cos(theta)), 3.0/2.0)

    return numerator/denominator


# Obtain the z-linspace
z_linspace = np.linspace(0, 4, 40+1)

# Instantiate the electric field arrays
E_own = [0]*len(z_linspace)
E_quad = [0]*len(z_linspace)

# Integrate for each z-value
for i in range(0, len(z_linspace)):
    # Set z
    z = z_linspace[i]

    # Calculate the electric field using my integrator (rectified integrand)
    E_own[i] = integrate_adaptive(integrand_func_rectified, 0, np.pi, 1e-7)

    # Calculates the electric field using quad (rectified integrand unnecessary)
    E_quad[i], err_ = scipy.integrate.quad(integrand_func, 0, np.pi)


# Multiply by the restrictivity and necessary constants
E_own = np.multiply(E_own, (rho*R*R)/(2*epsilon_0))
E_quad = np.multiply(E_quad, (rho*R*R)/(2*epsilon_0))
E_actual = calc_actual_spherical_E(z_linspace)


##### Showing the results #####

# Figure setup, Plots 
fig, ax = plt.subplots(1, 2, figsize=(16,9))
ax[0].plot(z_linspace, E_quad, color="red", marker="x")
ax[0].set_title("The Electric field through quad integration")
ax[0].set_ylabel("The Electric field")

err = np.abs(E_quad - E_actual)
ax[1].plot(z_linspace, err, color="red")
ax[1].set_title("The associated err (std = " + str(np.std(err)) + ")")
ax[0].set_ylabel("The error")
plt.show()


# Figure setup, Plots 
fig, ax = plt.subplots(1, 2, figsize=(16, 9))
ax[0].plot(z_linspace, E_own, color="red", marker="x")
ax[0].set_title("The Electric field through my own integration function")
ax[0].set_ylabel("The Electric field")

err = np.abs(E_own - E_actual)
ax[1].plot(z_linspace, err, color="red")
ax[1].set_title("The associated err (std = " + str(np.std(err)) + ")")
ax[0].set_ylabel("The error")
plt.show()

"""
Note: my own integration method does not accept the nan value at z=R and breaks down.
The quad method works at that point! The way I fixed this for my method (for display purposes)
was to use a rectified function where nan was avoided by setting the result to zero if the 
numerator was zero (bypassing 0/0). This works but has its issues.

z=R remains a weird point as both integration functions end up smoothing the discontinuity.
"""
