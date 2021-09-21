import numpy as np
import matplotlib.pyplot as plt
import sys

#sys.setrecursionlimit(10)

recursion_count = 0
function_call_count = 0

def simpson(func, a, b, f_a, f_interm, f_b):
    simpson_integral = ((b-a)/6.0)*(f_a + 4.0*f_interm + f_b)
    return  simpson_integral

def integrate_adaptive(func, a, b, tol, extra=None):  
    global recursion_count
    recursion_count += 1
    
     # extra[0:-2] is f_0, f_2, f_4
     # extra[-1] is previous integral
     # The first run
    if np.atleast_1d(extra).any() == None:    
        y = [0]*5
        for i in range(0, 5):
            y[i] = func(a + i*(b-a)/(len(y)-1))
        
        init_integral = simpson(func, a, b, y[0], y[2], y[4])
    else:
        y = [0]*5
        y[0] = extra[0]
        y[1] = func(a + 1*(b-a)/(len(y)-1))
        y[2] = extra[1]
        y[3] = func(a + 3*(b-a)/(len(y)-1))
        y[4] = extra[2]

        init_integral = extra[-1]
    
    finer_integral_L = simpson(func, a, (a+b)/2.0, y[0], y[1], y[2])
    finer_integral_R = simpson(func, (a+b)/2.0, b, y[2], y[3], y[4])

    finer_integral = finer_integral_L + finer_integral_R
        
    err = np.abs(init_integral - finer_integral)
    
    if err <= tol:
        return finer_integral
    
    else:
        extra_L = np.concatenate((y[0:3], [finer_integral_L]))
        extra_R = np.concatenate((y[2:], [finer_integral_R]))
            
        recursion_integral_L = integrate_adaptive(func, a,
                                                    (a+b)/2.0, tol, extra_L)           
        recursion_integral_R = integrate_adaptive(func, (a+b)/2.0,
                                                  b, tol, extra_R)
            
        return recursion_integral_L + recursion_integral_R
        

def lin_x(x):
    print("X:", x)
    global function_call_count
    function_call_count += 1
    return np.exp(x)

integ = integrate_adaptive(lin_x, -10, 10, 1.0e-7)

print("recursion count:", recursion_count)
print("function call count", function_call_count)
print(integ)
