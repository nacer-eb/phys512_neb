import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

# Setting up unit conversion variables
micro_s_in_s = 1e-6
minutes_in_s = 60
hours_in_s = 60*minutes_in_s
days_in_s = 24*hours_in_s
years_in_s = 365*days_in_s

# Defining the half lives 
half_lives = np.zeros(15) # in seconds
half_lives[0] = 4.468*1e9*years_in_s
half_lives[1] = 24.10*days_in_s
half_lives[2] = 6.7*hours_in_s
half_lives[3] = 245500*years_in_s
half_lives[4] = 75380*years_in_s
half_lives[5] = 1600*years_in_s
half_lives[6] = 3.8235*days_in_s
half_lives[7] = 3.10*minutes_in_s
half_lives[8] = 26.8*minutes_in_s
half_lives[9] = 19.9*minutes_in_s
half_lives[10] = 164.3*micro_s_in_s
half_lives[11] = 22.3*years_in_s
half_lives[12] = 5.015*years_in_s
half_lives[13] = 138.376*days_in_s
half_lives[14] = np.inf

def derivative_func(x, y):
    """
    The right hand side of the decay ODE. I call it the derivative of y.

    :param x: The independant variable (often time)
    :param y: The integrated function y at x. 

    :return: The derivative of y at x
    """
    # Sets up the array to hold the derivative values (15 total elements in the decay chain)
    dydx = np.zeros(15)

    # Calculate the production and degradation
    production = np.concatenate(([0], y[:-1]*(1.0/half_lives[:-1])))
    degradation = y*(1.0/half_lives)

    # Calculate dydx and return it
    dydx = production - degradation
    return dydx


# The initial concentration of each element
# Initially we start with U238 only
y0 = np.zeros(15)
y0[0] = 1

# The integration bounds (labelled x, but is time)
x0 = 0
x1 = 40*1e9*years_in_s

# Integrate (Using the Radau method for stiff equations)
# I ask for at least 100 points here, just to make the plot nicer.
ans = integrate.solve_ivp(derivative_func, [x0, x1], y0, method="Radau", max_step=((x1-x0)/20))

# Calculate the theoretical ratio between U238 and Pb204
x_dtl = np.linspace(x0, x1, 100)
y_thr = (1-np.exp(-x_dtl/half_lives[0]))/np.exp(-x_dtl/half_lives[0])

# Plot said ratio, and the theoretical  prediction
plt.scatter(ans.t/(1e9*years_in_s), ans.y[-1]/ans.y[0], color="red", label="Integrated values")
plt.plot(x_dtl/(1e9*years_in_s), y_thr, color="blue", label="Theoretical expectation")
plt.title("Ratio of Pb206 to Uranium238")
plt.xlabel("Billion years")
plt.ylabel("Ratio")
plt.legend()
plt.savefig("Ratio_Pb206_Ur238")
plt.cla()
plt.clf()
plt.close()


# Now for the second plot
# Set up the integration bounds
x0 = 0
x1 = 100*1e4*years_in_s

# Integrate (Using the Radau method for stiff equations)
ans = integrate.solve_ivp(derivative_func, [x0, x1], y0, method="Radau", max_step=((x1-x0)/20))

# Plot Th204/U234
plt.scatter(ans.t/(1e3*years_in_s), ans.y[4]/ans.y[3], color="red", label="Integrated values")
plt.title("Ratio of Uranium234 to Uranium238")
plt.xlabel("Thousand years")
plt.ylabel("Ratio")
plt.legend()
plt.savefig("Ratio_Th230_Ur234")
plt.cla()
plt.clf()
plt.close()

