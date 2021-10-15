import numpy as np
import matplotlib.pyplot as plt
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


def derivative_calc(f, f_m, m, i, h):
    dim = len(m)

    eps = np.zeros(dim)
    eps[i] = h

    d_i_f = (f(m+eps) - f_m)/(h)

    return d_i_f


def newton_step(f, y_true, N_inv, m, h_arr, step_size):
    dim_param = len(m)
    dim_pts = len(y_true)
    
    y = get_power_func(m)

    res = y_true - y[0:dim_pts]

    A_prime_T = np.zeros((dim_param, dim_pts))

    for i in range(0, dim_param):
        A_prime_T[i] = derivative_calc(f, y, m, i, h_arr[i])[0:dim_pts]

    print(A_prime_T)
    M_1 = (A_prime_T@N_inv)
    
    m_step = np.linalg.inv(M_1@A_prime_T.T)@(M_1@res)

    m_new = m + np.multiply(step_size, m_step)

    return m_new, A_prime_T


print("Theoretical Chi ", np.mean((0.5*(data[:, 2] + data[:, 3]))**2))

m = [69, 0.022, 0.12, 0.06, 2.1e-9, 0.95] #[60,0.02,0.1,0.05,2.00e-9,1.0]
h_arr = [0.01, 1e-5, 1e-4, 1e-5, 1e-12, 1e-4]

data_b = np.loadtxt("COM_PowerSpect_CMB-TT-binned_R3.01.txt", skiprows=1)


y = get_power_func(m)
y_true = data[:, 1]
res = y_true - y[0:len(data)]
chi =  res.T@N_inv@res
step_size = 1
for t in range(0, 20):
    m_new, A_prime_T = newton_step(get_power_func, data[:, 1], N_inv, m, h_arr, step_size)
    
    y = get_power_func(m_new)

    res = y_true - y[0:len(data)]
    chi_new = res.T@N_inv@res
    if chi_new < chi:
        chi = chi_new
        m = m_new
        step_size *= 2
        
    if chi_new >= chi:
        step_size *= 0.5
        
    
    
    print("Params after step", m_new, chi_new)
    print("Params chosen", m, chi)
    print("----------")

print("----------")
print("The final Params chosen", m, chi)
print("----------")

np.savetxt("opt_params.txt", m)


np.savetxt("opt_params_curvM.txt", A_prime_T@N_inv@A_prime_T.T)
