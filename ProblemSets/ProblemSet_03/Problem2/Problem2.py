import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

micro_s_in_s = 1e-6
minutes_in_s = 60
hours_in_s = 60*minutes_in_s
days_in_s = 24*hours_in_s
years_in_s = 365*days_in_s

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
half_lives[13] = 138.376*years_in_s
half_lives[14] = np.inf

def derivative_func(x, y):
    dydx = np.zeros(15)
    production = np.concatenate(([0], y[:-1]*(1.0/half_lives[:-1])))
    degradation = y*(1.0/half_lives)

    dydx = production - degradation
    return dydx


y0 = np.zeros(15)
y0[0] = 1

x0 = 0
x1 = 20.1*1e9*years_in_s

ans = integrate.solve_ivp(derivative_func, [x0, x1], y0, method="Radau")

for i in range(0, 15):
    plt.plot(ans.t, ans.y[i], label=str(i), marker="o")
    plt.legend()
    plt.show()


print(np.shape(ans.t[1]))
print(ans.nfev)
