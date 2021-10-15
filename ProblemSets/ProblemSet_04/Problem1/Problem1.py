import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
import camb

def get_power_func(param, lmax=3000):
    H0, ombh2, omch2, tau, As, ns = param[0:]
    camb_params = camb.CAMBparams()

    # Set camb parameters
    camb_params.set_cosmology(H0=H0, ombh2=ombh2,
                             omch2=omch2, mnu=0.06,
                             omk=0, tau=tau)
    camb_params.InitPower.set_params(As=As, ns=ns, r=0)
    camb_params.set_for_lmax(lmax, lens_potential_accuracy=0)

    camb_results = camb.get_results(camb_params)
    camb_power_spectras = camb_results.get_cmb_power_spectra(camb_params,
                                                     CMB_unit='muK')
    total_power = camb_power_spectras['total']
    TT = total_power[:, 0]

    return TT[2:]


data = np.loadtxt("COM_PowerSpect_CMB-TT-full_R3.01.txt", skiprows=1)

test_script_params = [60,0.02,0.1,0.05,2.00e-9,1.0]
new_params = [69, 0.022, 0.12, 0.06, 2.1e-9, 0.95]

data_TT = data[:, 1]
data_TT_err = 0.5*(data[:, 2] + data[:, 3])

test_script_TT = get_power_func(test_script_params)[0:len(data_TT)]
new_TT = get_power_func(new_params)[0:len(data_TT)]

test_script_TT_resid = data_TT - test_script_TT
new_TT_resid =  data_TT - new_TT

test_script_TT_chi_sqrd = np.sum(np.square(test_script_TT_resid/data_TT_err))
new_TT_chi_sqrd = np.sum(np.square(new_TT_resid/data_TT_err))

print("The test script chi²: %.2f" % test_script_TT_chi_sqrd)
print("The new paremeters' chi² %.2f:" % new_TT_chi_sqrd)

deg_freedom = len(data_TT) - len(new_params) # Same for both
print("With degrees of freedom:", deg_freedom)

chi_sigma = np.sqrt(deg_freedom)
print("Hence the second set of params is better",
      "than the first with a %.2lf sigmas of difference"
      % float((test_script_TT_chi_sqrd - new_TT_chi_sqrd)/chi_sigma))

data_binned = np.loadtxt("COM_PowerSpect_CMB-TT-binned_R3.01.txt", skiprows=1)

plt.scatter(data_binned[:, 0], data_binned[:, 1], color="red", s=2)
plt.plot(data[:, 0], test_script_TT, label="Test Script")
plt.plot(data[:, 0], new_TT, label="New params")
plt.legend()
plt.show()


