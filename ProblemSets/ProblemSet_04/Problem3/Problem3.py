import numpy as np
import numpy.random as random
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


#Taken from problem 2
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


def mcmc_step(f, y_true, N_inv, old_m, old_chi_sqrd, step_size, covariance):
    """
    Computes a single mcmc step
    
    :param f: The model function
    :param y_true: The actual data (Power spectra in our case)
    :param N_inv: The inverse of the error array of true data. (diag(err**-2))
    :param old_m: The previous model parameters.
    :param old_chi_sqrd: The previous chi² calculated
    :param step_size: The mcmc step size.
    :param covariance: The covariance between the parameters,
                       to estimate a better step size.

    :return: m_new, y_new
    """

    # Obtain the dimensions of the system
    dim_m = len(old_m)

    # Take a random step
    m_step = step_size*random.multivariate_normal(np.zeros(dim_m), covariance)

    # Obtain the new parameter set
    m = old_m + m_step

    # Calculate chi² and delta chi² 
    y, chi_sqrd = calc_chi_sqrd(f, m, y_true, N_inv)
    delta_chi_sqrd = chi_sqrd - old_chi_sqrd

    # If this step is good, try stepping further
    i = 1
    while delta_chi_sqrd < 0:
        # Stepping progressively further.
        i += 1
        new_m = m + m_step*1.2**i

        # Calculate the new_chi_sqrd and it's delta
        new_y, new_chi_sqrd = calc_chi_sqrd(f, new_m, y_true, N_inv)
        new_delta_chi_sqrd = new_chi_sqrd - old_chi_sqrd

        # If this is a worse improvement, don't use it
        if new_delta_chi_sqrd > delta_chi_sqrd:
            break

        # Else, if it is better (smaller new chi²) use it.
        m = new_m
        y = new_y

        # Save your new values and repeat the cycle until it stops improving
        chi_sqrd = new_chi_sqrd
        delta_chi_sqrd = new_delta_chi_sqrd

        # Verbosity is forced here - the program lasts a while
        print("The delta chi²", delta_chi_sqrd)
        
    # Calculate the probability to switch to the new m-state
    switch_prob = np.exp(-0.5*delta_chi_sqrd)

    # 'Roll the dice'
    rand_value = np.random.rand()

    # Verbose info.
    print("Delta", delta_chi_sqrd, "// RandValue", rand_value, "// SwitchProb", switch_prob)
    
    # If the switch is made.
    if rand_value <= switch_prob:
        return m, chi_sqrd

    # Else keep the old m-state
    return old_m, old_chi_sqrd


def mcmc_optimize(f, y_true, N_inv, m_init, covariance, step_size=1, steps=100):
    """
    Computes multiple mcmc steps and returns the chain; a list of [chi², m].
    
    :param f: The model function
    :param y_true: The actual data (Power spectra in our case)
    :param N_inv: The inverse of the error array of true data. (diag(err**-2))
    :param m_init: The initial model parameters.
    :param covariance: The covariance between the parameters,
                       to estimate a better step size.
    :param step_size: The mcmc step size.
    :param steps: The number of mcmc steps to compute.

    :return: None
    """

    # Start to setup the initial conditions
    y_init, init_chi_sqrd = calc_chi_sqrd(f, m_init, y_true, N_inv)

    print("The initial parameters are:", m_init)
    print("The initial chi² is:", init_chi_sqrd)

    # Setup the chain
    chain = np.zeros((steps, len(m_init)+1))
    
    # Setup the initial conditions.
    m = m_init
    chi_sqrd = init_chi_sqrd
    for t in range(0, steps):
        print("%d --------------------" %t)
        m, new_chi_sqrd = mcmc_step(f, y_true, N_inv, m, chi_sqrd, step_size, covariance)

        # If you are moving, try moving faster/further.
        if new_chi_sqrd != chi_sqrd:
            step_size = np.min((step_size*3.0/0.8, 10)) 

        # If you are not moving, make the steps smaller.
        if new_chi_sqrd == chi_sqrd:
            step_size *= 0.8

        # At convergence, the above will stabilize - or you add a **(1-i/steps) power to force step convergence

        # Then save the results
        chi_sqrd = new_chi_sqrd

        chain[t][0] = chi_sqrd
        chain[t][1:] = m

        print(m)
        print(" INITIAL CHI² %.4lf // CURRENT CHI² %.4lf" %(init_chi_sqrd, chi_sqrd))
        print("%d -------------------- Step_size: %lf" %(t, step_size))
        print(" ")

    return chain


# Obtain the planck data and error
data = np.loadtxt("COM_PowerSpect_CMB-TT-full_R3.01.txt", skiprows=1)
data_err = 0.5*(data[:, 2] + data[:, 3])
N_inv = np.diag(data_err**(-2))

# Obtain the initial parameters (I chose the ones from problem 1)
m_init = [69, 0.022, 0.12, 0.06, 2.1e-9, 0.95]
m_covariance = np.loadtxt("../Problem2/planck_fit_covariance.txt", delimiter=" ")

# Collect and save the chain
chain = mcmc_optimize(get_power_func, data[:, 1], N_inv, m_init, m_covariance, step_size=3, steps=2000)
np.savetxt("planck_chain.txt", chain)

# Getting and plotting the chain data
data_c = np.loadtxt("planck_chain.txt")

data_c_mean = np.mean(data_c[400:, 1:], axis=0)
data_c_std = np.std(data_c[400:, 1:], axis=0)
data_c_cov = np.cov(data_c[400:, 1:], rowvar=False)

print("The parameters are", data_c_mean, "//", np.shape(data_c_mean))
print(" ")
print("The standard deviations are", data_c_std)
print(" ")
print("The covariance matrix is", data_c_cov, "//", np.shape(data_c_cov))

fig, ax = plt.subplots(3, 2, figsize=(16, 9))

index = 1
for i in range(0, 3):
    for j in range(0, 2):
        ft = np.abs(np.fft.rfft(data_c[100:, index]))
        ax[i][j].loglog(np.full(len(ft), np.mean(ft[1: 50])), color="red")
        ax[i][j].loglog(ft)
        index += 1
plt.savefig("FourierTransform")
plt.cla()
plt.clf()
plt.close()


    
