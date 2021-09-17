import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import warnings

warnings.filterwarnings('ignore')

def getSamplePoints(x, x_data, y_data, npts):
    # Changes a constant with an array with size 1, and does nothing to arrays
    x_arr = np.atleast_1d(np.multiply(x, 1.0))

    # Make sure you are within range
    assert (x_arr >= np.min(x_data)).all()
    assert (x_arr <= np.max(x_data)).all()

    # Look for the closest indices of each v_arr[i]
    closest_indices = np.zeros(len(x_arr), dtype=int)
    for i in range(len(x_arr)):
        closest_indices[i] = np.where(x_arr[i] >= x_data)[0][-1]

    # Safeguard against being at the boundary
    closest_indices = np.where(closest_indices <= int(npts/2)-1, int(npts/2)-1, closest_indices)
    closest_indices = np.where(closest_indices >= len(x_data)-int(npts/2)-1,
                               len(x_data)-int(npts/2)-1, closest_indices)

    
    # Create the index at which you want your sample points, assuming symmetric sampling.
    index_range = [0]*npts
    for i in range(0, int(npts)):
        index_range[i] = closest_indices+(i-int(npts/2)+1)
  

    # Take each of the sample points and put them into an array. [x-index][sample point index]
    x_sample = np.take(x_data, index_range).T
    y_sample = np.take(y_data, index_range).T
    
    return x_sample, y_sample


def calcRational(x, p, q):
    # Turn any constant into a 1D array
    x = np.atleast_1d(x)
    
    num = np.zeros(len(x))
    denum = np.ones(len(x))
    
    for n in range(0, len(p)):
        num += p[n]*np.power(x, n)
    
    for m in range(0, len(q)):
        denum += q[m]*np.power(x, m+1) # Recall we start with x¹ not x⁰
    return np.divide(num, denum)


def fitRational(x, y, n, m, pinv=False): # n and m are size of p and q NOT orders
    assert n+m == len(x) # make sure we have an invertible matrix
    combined_param = np.zeros(n+m)
    new_x = np.zeros((len(x), n+m))

    # I went for slower but more readable code using the loops here
    # I might change it after
    for i in range(len(x)):
        for j in range(n):
            new_x[i][j] = x[i]**j
        for j in range(n, m+n):
            new_x[i][j] = -y[i]*x[i]**(j+1-n)

    if not pinv:
        inv_new_x = np.linalg.inv(new_x)

    if pinv:
        inv_new_x = np.linalg.pinv(new_x)

   
    combined_param = np.dot(inv_new_x, y)

    p = combined_param[0:n]
    q = combined_param[n:]
    
    return p,q


def interpolate(x, x_data, y_data, type, params):
    # 0 - Polynomial with numpy polyfit: params [npts, deg]
    # 1 - Polynomial with inverse fit (using ratfit): params [npts]
    # 2 - Cubic spline with scipy fit: params [npts]
    # 3 - Rational fit with inv: params [npts, n, m]
    # 4 - Rational fit with pinv: params [npts, n, m]

    # Turn any constant into a 1D array
    x_arr = np.atleast_1d(np.multiply(x, 1.0))
    npts = params[0]
    
    y = np.zeros(len(x_arr))
    x_sample, y_sample = getSamplePoints(x, x_data, y_data, npts) # Get sample points
    for i in range(len(x_arr)):
        if type == 0:
            deg = params[1]
            p = np.polyfit(x_sample[i], y_sample[i], deg)
            y[i] = np.polyval(p, x_arr[i])
            
        if type == 1:
            deg = params[0] # Necessary for invertibilty
            p, q = fitRational(x_sample[i], y_sample[i], deg, 0)
            y[i] = calcRational(x_arr[i], p, q)
            
        if type == 2:
            cs = CubicSpline(x_sample[i], y_sample[i])
            y[i] = cs(x_arr[i])
            
        if type == 3:
            n = params[1]
            m = params[2]
            p, q = fitRational(x_sample[i], y_sample[i], n, m)
            y[i] = calcRational(x_arr[i], p, q)
            
        if type == 4:
            n = params[1]
            m = params[2]
            p, q = fitRational(x_sample[i], y_sample[i], n, m, True)
            y[i] = calcRational(x_arr[i], p, q)

    if type >= 3:
        return p, q, y
    return y


######### Our comparison function ############3

def compareFits(func, sample_x, x, text, deg, n, m):
    # Get the rest of the data
    sample_y = func(sample_x)
    y = func(x)
    
    # Interpolate using polynomials with the polyfit 4 pts 4 deg (deg can be changed)
    y0 = interpolate(x, sample_x, sample_y, 0, [deg, deg])

    # Interpolate using polynomials by taking the inv. 4pts (needs 4deg)
    y1 = interpolate(x, sample_x, sample_y, 1, [deg, deg])
    
    # Interpolate with Cubic Spline 4pts
    y2 = interpolate(x, sample_x, sample_y, 2, [deg])
    
    # Interpolate with a Rational function (using inv) 4deg, n=2, m=2
    p3, q3, y3 = interpolate(x, sample_x, sample_y, 3, [deg, n, m])
    
    # Interpolate with a Rational function (using pinv) 4deg, n=2, m=2
    p4, q4, y4 = interpolate(x, sample_x, sample_y, 4, [deg, n, m])
    
    cmap = matplotlib.cm.get_cmap('Dark2')
    fig, ax = plt.subplots(2, 3, figsize=(16, 12))
    
    ax[0, 0].plot(x, np.abs(y0 - y), color=cmap(0))
    ax[0, 0].set_ylabel("The Error")
    ax[0, 0].set_title("a) Polynomial + Polyfit")
    
    ax[0, 1].plot(x, np.abs(y1 - y), color=cmap(1))
    ax[0, 1].set_ylabel("The Error")
    ax[0, 1].set_title("b) Polynomial + Inverse")
    
    ax[0, 2].plot(x, np.abs(y2 - y), color=cmap(2))
    ax[0, 2].set_ylabel("The Error")
    ax[0, 2].set_title("c) Cubic Spline")
    
    ax[1, 0].plot(x, np.abs(y3 - y), color=cmap(3))
    ax[1, 0].set_ylabel("The Error")
    ax[1, 0].set_title("d) Rational Function with Inv")
    
    ax[1, 1].plot(x, np.abs(y4 - y), color=cmap(4))
    ax[1, 1].set_ylabel("The Error")
    ax[1, 1].set_title("e) Rational Function with PInv")
    
    ax[1, 2].text(-0.1, 0.50, text, color='black', fontsize=12, wrap=True, va='center')
    ax[1, 2].axis('off')
    plt.show()

    return p3, q3, p4, q4


# First for the cosine
sample_x_cos = np.linspace(-np.pi, np.pi, 10)
x_cos = np.linspace(np.min(sample_x_cos), np.max(sample_x_cos), 100)

caption_cos = "As you can see the error varies a lot. I added a few more fit options, because I was curious."\
        +" Notice how much better (b) and (c) do compared to (a). This is especially interesting when comparing (a)"\
        +" and (b) as only the method of fitting changes not the fit function."\
        +" The rational function fits slightly worse than (a), independently of the use of inv or pinv"

compareFits(np.cos, sample_x_cos, x_cos, caption_cos, 4, 2, 2)


# Then for 1/(1+x²)

def inverseQuadratic(x):
    x = x 
    return 1.0/(1.0+x*x)

sample_x_iq = np.linspace(-1, 1, 20)
x_iq = np.linspace(np.min(sample_x_iq), np.max(sample_x_iq), 100)

# Interpolate with a Rational function (using inv) 4deg, n=2, m=2
p3, q3, y3 = interpolate(x_iq, sample_x_iq, inverseQuadratic(sample_x_iq), 3, [10, 4, 6])

# Interpolate with a Rational function (using pinv) 4deg, n=2, m=2
p4, q4, y4 = interpolate(x_iq, sample_x_iq, inverseQuadratic(sample_x_iq), 4, [10, 4, 6])

caption_iq =  "Again, the error varies a lot. \n"\
    + "The (b) again does much better than (a) with the Cubic Spline trailing behind \n"\
    + "The rational function now gives us interesting information, using pinnv leads to a much better fit than Inv,"\
    + " the former being the best fit by far. \n"\
    + "Let's look at the parameters\n"\
    + "For the inv, p: " + '{0:.2f}, '.format(p3[0]) + '{0:.2f}, '.format(p3[1]) + '{0:.2f}, '.format(p3[2])\
    + '{0:.2f} '.format(p3[3]) + "\n"\
    + "For the inv, q: " + '{0:.2f}, '.format(q3[0]) + '{0:.2f}, '.format(q3[1]) + '{0:.2f}, '.format(q3[2])\
    + '{0:.2f}, '.format(q3[3]) + '{0:.2f} '.format(q3[4]) + '{0:.2f} '.format(q3[5]) + "\n"\
    + "\n"\
    + "For the pinv, p: " + '{0:.2f}, '.format(p4[0]) + '{0:.2f}, '.format(p4[1]) + '{0:.2f}, '.format(p4[2])\
    + '{0:.2f}, '.format(p4[3]) + "\n"\
    + "For the pinv, q: " + '{0:.2f}, '.format(q4[0]) + '{0:.2f} '.format(q4[1]) + '{0:.2f}, '.format(q4[2])\
    + '{0:.2f}, '.format(q4[3]) + '{0:.2f} '.format(q4[4]) + '{0:.2f} '.format(q4[5]) + "\n"\
    + "The idea is that for large enough n,m since abs(x) < 1 the powers of x tend to get very small, in return the inverse"\
    + " blows up to large values. In effect this often causes numerator to get non-zero x², x³ -> leading to weird denominators"\
    + " pinv essentially eliminates these values, by setting them to zero in the inverse (1/x -> 0) instead of very large"\
    + " which from my understanding is the equivalent of 'ignoring' these troublesome values. It's also important to note"\
    + " that inv also leads to denominators which go to zero in the region of interest, which is what we wanted to avoid"\
    + " when we wrote it as 1+.. and fixing the 1."


p3, q3, p4, q4 = compareFits(inverseQuadratic, sample_x_iq, x_iq, caption_iq, 10, 4, 6)

