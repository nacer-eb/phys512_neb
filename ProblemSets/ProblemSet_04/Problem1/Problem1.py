import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
import camb

def get_power_func(param, lmax=3000):
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


# Load the full planck data, spherical harmonics
data = np.loadtxt("COM_PowerSpect_CMB-TT-full_R3.01.txt", skiprows=1)

# Loads the planck data power spectra, dim and errors
data_TT = data[:, 1]
data_TT_dim = len(data_TT)
data_TT_err = 0.5*(data[:, 2] + data[:, 3])

# The parameter sets to compare
test_script_params = [60,0.02,0.1,0.05,2.00e-9,1.0]
new_params = [69, 0.022, 0.12, 0.06, 2.1e-9, 0.95]

# Obtain the power spectra for each parameter sets
test_script_TT = get_power_func(test_script_params, data_TT_dim)
new_TT = get_power_func(new_params, data_TT_dim)

# Calculate the residuals for each of the parameter sets
test_script_TT_resid = data_TT - test_script_TT
new_TT_resid =  data_TT - new_TT

# Calculate chi²
test_script_TT_chi_sqrd = np.sum(np.square(test_script_TT_resid/data_TT_err))
new_TT_chi_sqrd = np.sum(np.square(new_TT_resid/data_TT_err))

print("The test script chi²: %.2f" % test_script_TT_chi_sqrd)
print("The new paremeters' chi² %.2f:" % new_TT_chi_sqrd)

# Compute the degrees of freedom of our planck data
deg_freedom = len(data_TT) - len(new_params) # Same for both
print("With degrees of freedom/mean chi²:", deg_freedom)

# Obatin the std in the expected chi²
chi_sigma = np.sqrt(2*deg_freedom)
print("The sigma in chi² is: %.2lf" % chi_sigma)
print("Hence the test script's leads to a chi² %.2lf sigmas away from mean."
      % float((test_script_TT_chi_sqrd - deg_freedom)/chi_sigma))
print("And the new set of params' chi² is %.2lf sigmas away from mean."
      % float((new_TT_chi_sqrd - deg_freedom)/chi_sigma))

# Obatined a reduced form of the data for plotting.
data_binned = np.loadtxt("COM_PowerSpect_CMB-TT-binned_R3.01.txt", skiprows=1)

# Plot everything
fig, ax = plt.subplots(1, 1, figsize=(16, 9))
plt.errorbar(data_binned[:, 0], data_binned[:, 1], 0.5*(data_binned[:, 2] + data_binned[:, 3]), fmt="ro", markersize=2)
plt.plot(data[:, 0], test_script_TT, label="Test Script")
plt.plot(data[:, 0], new_TT, label="New params")
plt.legend()
plt.savefig("ParameterSet_Comparison")
plt.cla()
plt.clf()
plt.close()


fig, ax = plt.subplots(1, 1, figsize=(16, 9))
plt.plot([data[0, 0], data[-1, 0]], [1, 1], color="red")
plt.scatter(data[:, 0], np.abs(new_TT-data[:, 1])/np.abs(data_TT_err), s=1)
plt.title("Error analysis of new params")
plt.ylabel("$\sigma$ away from data")
plt.xlabel("Data points")
plt.tight_layout()
plt.savefig("NewSet_Residuals")
plt.cla()
plt.clf()
plt.close()

fig, ax = plt.subplots(1, 1, figsize=(16, 9))
plt.plot([data[0, 0], data[-1, 0]], [1, 1], color="red")
plt.scatter(data[:, 0], np.abs(test_script_TT-data[:, 1])/np.abs(data_TT_err), s=1)
plt.title("Error analysis of test script params")
plt.ylabel("$\sigma$ away from data")
plt.xlabel("Data points")
plt.tight_layout()
plt.savefig("TestScript_Residuals")
plt.cla()
plt.clf()
plt.close()
