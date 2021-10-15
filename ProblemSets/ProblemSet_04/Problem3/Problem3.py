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


def mcmc_step(f, y_true, N_inv, m, old_chi_sqrd, step_size, curvM):
    dim_m = len(m)
    
    m_step = 0.01*step_size*np.random.multivariate_normal(np.zeros(dim_m),
                                                     np.linalg.inv(curvM))

    print(m_step)
    #np.linalg.inv(curvM)@np.random.randn(dim_m)
    new_m = m + m_step
    y = f(new_m)
 
    res = y_true - y[0: len(y_true)]

    new_chi_sqrd = res.T@N_inv@res
    delta_chi_sqrd = new_chi_sqrd - old_chi_sqrd

    i = 1
    while delta_chi_sqrd < 0:
        i += 1
        new_new_m = new_m + m_step*1.2**i
        new_y = f(new_new_m)
 
        new_res = y_true - new_y[0: len(y_true)]

        new_new_chi_sqrd = new_res.T@N_inv@new_res
        new_delta_chi_sqrd = new_new_chi_sqrd - old_chi_sqrd

        if new_delta_chi_sqrd > delta_chi_sqrd:
            break
        new_m = new_new_m
        y = new_y

        new_chi_sqrd = new_new_chi_sqrd
        delta_chi_sqrd = new_delta_chi_sqrd
        print(delta_chi_sqrd)
        
    switch_prob = np.exp(-0.5*delta_chi_sqrd)

    rand_value = np.random.rand()

    print("delta", delta_chi_sqrd)
    print("rand_value", rand_value)
    print("switch prob", switch_prob)
    
    if rand_value <= switch_prob:
        return new_m, new_chi_sqrd

    # Else
    return m, old_chi_sqrd


opt_param = np.loadtxt("../Problem2/opt_params.txt")
opt_param = np.multiply(opt_param, 1 + 0.0520*np.random.rand(len(opt_param)) )
opt_param_curvM = np.loadtxt("../Problem2/opt_params_curvM.txt", delimiter=" ")

data = np.loadtxt("COM_PowerSpect_CMB-TT-full_R3.01.txt", skiprows=1)
N_inv = np.diag((0.5*(data[:, 2] + data[:, 3]))**(-2))


y = get_power_func(opt_param)

res = data[:, 1] - y[0: len(data)]

print(np.sum(np.square(res/(0.5*(data[:, 2] + data[:, 3])))))

opt_param_chi = res.T@N_inv@res

print("Initial Chi", opt_param_chi)

m, chi_sqrd = opt_param, opt_param_chi

chain_length = 500

m_list = np.zeros((chain_length, len(m)+1))

step_size = 1000
"""    
for t in range(0, chain_length):
    print(t)

    m, chi_sqrd_new = mcmc_step(get_power_func, data[:, 1],
                            N_inv, m, chi_sqrd,
                       step_size, opt_param_curvM)


    if chi_sqrd_new != chi_sqrd:
        step_size *= 1.0/0.7
        
    if chi_sqrd_new == chi_sqrd:
        step_size *= 0.7

    chi_sqrd = chi_sqrd_new
        
    m_list[t][0] = chi_sqrd
    m_list[t][1:] = m
    print(m)
    print(opt_param_chi, chi_sqrd)

    print("--------------------",step_size)

np.savetxt("chain.txt", m_list)

for s in range(0, chain_length):
    plt.plot(m_list.T[s])
    plt.show()
"""
chain = np.loadtxt("chain.txt")

for i in range(0, len(m)+1):
    plt.plot(chain[200:,i])
    plt.show()

    plt.loglog(np.abs(np.fft.rfft(chain[200:,i])))
    plt.show()

    print(np.mean(chain[200:, i]))
    print(np.std(chain[200:, i]))
