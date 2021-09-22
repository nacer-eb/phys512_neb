import numpy as np
import numpy.polynomial.chebyshev as cheb
import matplotlib.pyplot as plt


def mylog2_fit(deg, tol):
    x_true = np.linspace(0.5, 1, 100)
    y_true = np.log2(x_true)
    
    coef, res = cheb.chebfit(x_true, y_true, deg, full=True)

    if res[0][0] > tol:
        return mylog2_fit(deg+1, tol)
    
    return coef


def mylog2(x, coef):
    m,expo = np.frexp(x)
    nat_log = (expo + cheb.chebval(m, coef))/np.log2(np.e)

    return nat_log


x_space = np.linspace(1e-7, 100, 10000)
coef = mylog2_fit(3, 1e-15)

y = mylog2(x_space, coef)
y_true = np.log(x_space)

plt.plot(x_space, y)
plt.show()

plt.scatter(x_space, np.abs(y-y_true), s=4)
plt.show()

print(np.max(np.abs(y-y_true)))
