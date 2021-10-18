import numpy as np
import matplotlib.pyplot as plt
import camb

# Taken from Problem 1
def get_power_func(param, lmax=2507):
    """
    Calculates the TT power spectra using the given params
    
    :param param: Array containing the following  parameters in order: H_0, 
                  Ombh², Omch², Tau, A_s, n_s.

    :return: The TT power spectra values from l=2 up to lmax+2.
    """
    # Transfer params from array to variables
    H0, ombh2, omch2, tau, As, ns = param[0:]

    # Instanciate the camb parameter class
    camb_params = camb.CAMBparams()

    # Set camb parameters
    camb_params.set_cosmology(H0=H0, ombh2=ombh2,
                             omch2=omch2, mnu=0.06,
                             omk=0, tau=tau)
    camb_params.InitPower.set_params(As=As, ns=ns, r=0)
    camb_params.set_for_lmax(lmax, lens_potential_accuracy=0)

    # Obtain the parameter results & set units
    camb_results = camb.get_results(camb_params)
    camb_power_spectra = camb_results.get_cmb_power_spectra(camb_params,
                                                     CMB_unit='muK')

    # Obtain total power from the power spectra results
    total_power = camb_power_spectra['total']
    TT = total_power[:, 0]

    return TT[2:lmax+2]


def derivative_calc(f, f_m, m, i, h):
    """
    Calculates the derivative of f(m): df/dm_i 
    
    :param f: The function whose derivative to calculate
    :param f_m: The 'initial' y-value f(m) to avoid recalculating it
              Saves 1 function call, negligeable effect on precision.
    :param m: The point-m at which to calculate the derivative
    :param i: The direction in which to take the derivative. (f: R^N -> R, m is N dimensional)
    :param h: The step size; The h in (f(m+h)-f(m))/h

    :return: The derivative of f with input m.
    """
    # Obtain the dimensionality of m
    dim = len(m)

    # Setup the step size in direction m_i
    eps = np.zeros(dim)
    eps[i] = h

    # Take the derivative - Minimizing function calls.
    d_i_f = (f(m+eps) - f_m)/(h)

    return d_i_f


def newton_step(f, y_true, y, N_inv, m, h_arr, step_size):
    """
    Computes a single newton optimization step.
    
    :param f: The model function
    :param y_true: The true y_data (Power spectra in our case)
    :param y: The previous step's f(m) - to avoid repeating function calls
    :param N_inv: The inverse of the error array of true data. (diag(err**-2))
    :param m: The current model parameters. 
    :param h_arr: The derivative step sizes in each direction.
    :param step_size: The Newton step sizes for this optimization step.

    :return: The new parameters, the 'gradients' of f.
    """
    # Getting our dimensionality parameters
    dim_param = len(m)
    dim_pts = len(y_true)

    # Calculating the gradients
    A_prime_T = np.zeros((dim_param, dim_pts))
    for i in range(0, dim_param):
        A_prime_T[i] = derivative_calc(f, y, m, i, h_arr[i])
    
    # Computing the residuals
    res = y_true - y

    # Computing a matrix to be re-used many times in the next step. Saves compute time.
    M_1 = (A_prime_T@N_inv) 

    # Computing the Newton optimization step - assuming our simplificiation on the curvature of f.
    m_step = np.linalg.inv(M_1@A_prime_T.T)@(M_1@res)

    # Obtain our new parameter set m.
    m_new = m + np.multiply(step_size, m_step)

    return m_new, A_prime_T


def calc_chi_sqrd(f, m, y_true, N_inv):
    """
    Calculates chi².
    
    :param f: The model function
    :param m: The current model parameters.
    :param y_true: The true y_data (Power spectra in our case)
    :param N_inv: The inverse of the error array of true data. (diag(err**-2))

    :return: f(m), chi²
    """
    # Calulate f(m), then the residuals, then chi²
    y = f(m)
    res = y_true - y
    chi_sqrd = res.T@N_inv@res

    return y, chi_sqrd


def newton_optimize(f, y_true, N_inv, m = None, h_arr = None, step_size=1, steps = 10, verbose=False):
    """
    Computes multiple Newton's method steps.
    
    :param f: The model function
    :param y_true: The true y_data (Power spectra in our case)
    :param N_inv: The inverse of the error array of true data. (diag(err**-2))
    :param m: The current model parameters. (Has a default value - from problem 1)
    :param h_arr: The derivative step sizes in each direction. (Has a default value - 3 orders of magnitude below m)
    :param step_size: The Newton step sizes for this optimization step. (By default 1)
    :param steps: The number of optimization steps. (By default 10)
    :param verbose: Whether or not to print out the progress step by step.

    :return: m, chi², A_prime_T
    """
    if m is None:
        m = [69, 0.022, 0.12, 0.06, 2.1e-9, 0.95] # default m
    if h_arr is None:
        h_arr = [0.01, 1e-5, 1e-4, 1e-5, 1e-12, 1e-4] # default h_arr

    # Calculate the initial y and chi²
    y, chi = calc_chi_sqrd(f, m, y_true, N_inv)
    for t in range(0, steps):
        # Take a single newton optimization step
        m_new, A_prime_T = newton_step(f, y_true, y, N_inv, m, h_arr, step_size)

        # Calculate the new y, chi²
        y_new, chi_new = calc_chi_sqrd(f, m_new, y_true, N_inv) 

        # If chi² indicates an improvement take larger steps and update m=m_new etc
        if chi_new < chi:
            m = m_new
            y = y_new
            chi = chi_new
            step_size *= 2
            
        # Newton's method might've stepped too far. The new parameters are worse, try again with a smaller step size.
        if chi_new > chi:
            step_size *= 0.5

        # If verbose, this will keep you posted on the progress
        if verbose:
            print("Step %d, chi² %.2lf" %(t, chi))

    return m, chi, A_prime_T

# Load the power spectra data
data = np.loadtxt("COM_PowerSpect_CMB-TT-full_R3.01.txt", skiprows=1)

# Compute the error in the data
data_err = 0.5*(data[:, 2] + data[:, 3])
N_inv = np.diag(data_err**(-2))

# Start the Newton optimizer. Faster, but starts from closer.
# m, chi, A_prime_T = newton_optimize(get_power_func, data[:, 1], N_inv, steps = 5, verbose=True)

# If you want to start from the Test Script values uncomment the line below and comment the line above. Slower but starts from further
m, chi, A_prime_T = newton_optimize(get_power_func, data[:, 1], N_inv, m=[60,0.02,0.1,0.05,2.00e-9,1.0], step_size=0.1, steps = 20, verbose=True)

# Print out the final parameters
print("----------")
print("The final Params chosen", m, chi)
print("----------")

# Save the parameters and the estimate for the covariance matrix.
np.savetxt("planck_fit_params.txt", m)
np.savetxt("planck_fit_covariance.txt", np.linalg.inv(A_prime_T@N_inv@A_prime_T.T))

# Plotting Data
best_fit_params = np.loadtxt("planck_fit_params.txt")
best_fit_param_cov = np.loadtxt("planck_fit_covariance.txt")

print("----------")
print("The standard deviations are:", np.sqrt(np.diag(best_fit_param_cov)))
print("----------")

p1_params = [69, 0.022, 0.12, 0.06, 2.1e-9, 0.95]
data_b = np.loadtxt("COM_PowerSpect_CMB-TT-binned_R3.01.txt", skiprows=1)


fig, ax = plt.subplots(1, 1, figsize=(16, 9))
plt.hist(np.abs((get_power_func(best_fit_params)-data[0:, 1])/data_err), bins=15, density=True, alpha=0.5, label="The best fit")
plt.hist(np.abs((get_power_func(p1_params)-data[0:, 1])/data_err), bins=15, density=True, alpha=0.5, label="P1 Params")
plt.xlabel("Data-model error in multiple of $\sigma$")
plt.ylabel("Number of points")
plt.title("The errors in the Newton-optimized model and the model-params of problem 1")
plt.legend()
plt.savefig("Error_histogram")
plt.clf()
plt.cla()
plt.close()

