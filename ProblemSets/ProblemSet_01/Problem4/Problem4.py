import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import warnings

# This is for polyfit, it warns you when you specify ndeg=npts.
#But that's what I want, to compare the two methods of fitting (least# square and inv.)
warnings.filterwarnings('ignore') 

def getSamplePoints(x, x_data, y_data, npts):
    """
    Obtains npts-amount of sample points from x_data, y_data centered at x
    (Ex: x: 2, x_data: [0, 1, 2, 3, 4, 5], npts:4, x_sample=>[1, 2, 3, 4])
    (Ex2: x: 2.5, x_data: [0, 1, 2, 3, 4, 5], npts:4, x_sample=>[1, 2, 3, 4])
    
    :param x: The input about which sample points are chosen (can be an array
              or scalar - arrays lead to arrays of scalar points, 
              x[i]'s treated seperately)
    :param x_data: The general x_data from which to choose x_sample points
    :param y_data: The general y_data from which to choose y_sample points
    :param npts: The number of sample points

    :return: x_sample points and y_sample points
    """
    
    # Changes a constant with an array with size 1, and does nothing to arrays
    x_arr = np.atleast_1d(np.multiply(x, 1.0))

    # Make sure you are within range of the your gen. data
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
    """
    Calculates the rational function of x given p, q parameters. 
    
    :param x: The input independant variable (can be scalar or array)
    :param p: The numerator coefficients (1d array)
    :param q: The denominator coefficients (1d array)

    :return: The y-value for each x. (y is the same shape as x)
    """
    # Turn any constant into a 1D array
    x = np.atleast_1d(x)

    # Array that will hold our numerators for each x[i]
    num = np.zeros(len(x))

    # Array that will hold our denominator for each x[i] (Denum inits at 1)
    denum = np.ones(len(x))

    # Computes numerators as p[0] + p[1]*x + p[2]*x² + ...
    for n in range(0, len(p)):
        num += p[n]*np.power(x, n)

    # Computes denominators as 1 + q[0]*x + q[1]*x² + ...
    for m in range(0, len(q)):
        denum += q[m]*np.power(x, m+1) # Recall we start with x¹ not x⁰
    return np.divide(num, denum)


def fitRational(x, y, n, m, pinv=False): # n and m are size of p and q NOT orders
    """
    Fits a functional function to the x,y data. The numerator has n TERMS and the denominator 
    has m TERMS (excluding the +1 - so m-1 total terms). 
    
    :param x: The independant variable for the fit (1d array)
    :param y: The dependante variable for the fit (1d array)
    :param n: The number of numerator coefficients (scalar)
    :param m: The number of denominator coefficients (scalar)
    :param pinv: User-input, whether to use pinv or not 
                 (Boolean: By default False, does not use pinv instead uses inv)

    :return: p, q coefficients in 1d arrays of size n and m respecitvely.
    """
    # Make sure we have a square matrix (Necessary for invertibility)
    assert n+m == len(x)

    # Create an array to store our parameters (v-stacked p and q)
    combined_param = np.zeros(n+m)

    # Create an array to store our modified x (see the math pdf) which allows us to v-stack p and q simply
    new_x = np.zeros((len(x), n+m))

    # I went for slower but more readable code using the loops here
    # Computes our new_x as seen in the math pdf (included in the github repo)
    for i in range(len(x)):
        for j in range(n):
            new_x[i][j] = x[i]**j
        for j in range(n, m+n):
            new_x[i][j] = -y[i]*x[i]**(j+1-n)

    # Inverts the new_x matrix
    # Here the user (by default) does not want to use the pinv function
    if not pinv:
        inv_new_x = np.linalg.inv(new_x)

    # Inverts the new_x matrix
    # Here the user wants to use the pinv function
    if pinv:
        inv_new_x = np.linalg.pinv(new_x)

    # Use matrix multiplication to obtained or (v-stacked) parameters/coefficients
    combined_param = np.dot(inv_new_x, y)

    # Seperate the numerator and denominator parameters/coefficients from the v-stack
    p = combined_param[0:n]
    q = combined_param[n:]

    # return the parameters
    return p,q


def interpolate(x, x_data, y_data, type, params):
    """
    Interpolates x, using the x_data and y_data. 
    
        
    :param x: The independant variable of the interpolation (1d array or scalar)

    :param x_data: The x data used to generate fits and hence interpolation (1d array)
    :param y_data: The y data used to generate fits and hence interpolation (1d array)

    :param type: Specifies interpolation method and param structure (see below)
    :param params: The parameters of the interpolation (1d array)
          
    Type:
         0 - Polynomial with numpy polyfit. params [npts, deg]
         1 - Polynomial with inverse fit (using ratfit). params: [npts]
         2 - Cubic spline with scipy fit. params: [npts]
         3 - Rational fit with inv. params: [npts, n, m]
         4 - Rational fit with pinv. params: [npts, n, m] 

    :return: p, q coefficients when relevant and always the interpolated y data.
    """
    # Turn any constant into a 1D array
    x_arr = np.atleast_1d(np.multiply(x, 1.0))

    #Obtains the number of points to sample/use for the fit for each x[i]
    #SAME SAMPLING FOR EACH METHOD - So we can judge methods and avoid confounding variables
    npts = params[0]

    # Create a y-array to store or interpolated values from the fit.
    y = np.zeros(len(x_arr))

    # Obtain the sample points (2d array; done for each x) to use in the fit and hence in the interpolation
    x_sample, y_sample = getSamplePoints(x, x_data, y_data, npts) # Get sample points

    # For each x[i] we want to obtain the fit using the x,y_sample[i] and interpolate the data at x[i]
    for i in range(len(x_arr)):
        
        # 0 - Polynomial with numpy polyfit: params: [npts, deg]
        if type == 0: 
            # Obtain the number of degrees for the polynomial
            deg = params[1]

            # Obtains the polynomial coefficients
            p = np.polyfit(x_sample[i], y_sample[i], deg)

            # Evaluates the polynomial at x[i] to obtain the interpolated y[i]
            y[i] = np.polyval(p, x_arr[i])
        
        # 1 - Polynomial with inverse fit (using ratfit): params [npts]
        if type == 1:
            # Obtain the number of degrees (= the number of points)
            deg = params[0] # Necessary for invertibilty (square matrix requirement)

            # Obtain the p (and empty q) coefficients
            p, q = fitRational(x_sample[i], y_sample[i], deg, 0)

            # Evaluate the polynomial (Rational with denominator = 1) to obtain the interpolated data
            y[i] = calcRational(x_arr[i], p, q)
        
        # 2 - Cubic spline with scipy fit: params [npts]
        if type == 2:
            # Fit a cubic spline
            cs = CubicSpline(x_sample[i], y_sample[i])

            # Evalutate at the x[i]
            y[i] = cs(x_arr[i])

        # 3 - Rational fit with inv: params [npts, n, m]
        if type == 3:
            # Obtain n,m from the param array
            n = params[1]
            m = params[2]

            # Fit the rational func (with inv), obtain p,q.
            p, q = fitRational(x_sample[i], y_sample[i], n, m)

            # Evaluate at x[i]
            y[i] = calcRational(x_arr[i], p, q)

        # 4 - Rational fit with pinv: params [npts, n, m]
        if type == 4:
            # Obtain n,m from the param array
            n = params[1]
            m = params[2]

            # Fit the rational func (with pinv), obtain p,q.
            p, q = fitRational(x_sample[i], y_sample[i], n, m, True)

            # Evaluate at x[i]
            y[i] = calcRational(x_arr[i], p, q)

    # Specify for futur use that you want the rational coef. when rational funcs. are used.
    if type >= 3:
        return p, q, y

    # Else only return the intepolated data
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

    # Instantiate a cmap to add some colors to our plots.
    cmap = matplotlib.cm.get_cmap('Dark2')

    # Instantiate the figure and axes
    fig, ax = plt.subplots(2, 3, figsize=(16, 12))

    # Plot the errors for each fit
    # Polynomial + Polyfit
    ax[0, 0].plot(x, np.abs(y0 - y), color=cmap(0))
    ax[0, 0].set_ylabel("The Error")
    ax[0, 0].set_title("a) Polynomial + Polyfit")

    # Polynomial + Inverse
    ax[0, 1].plot(x, np.abs(y1 - y), color=cmap(1))
    ax[0, 1].set_ylabel("The Error")
    ax[0, 1].set_title("b) Polynomial + Inverse")

    # Cubic Spline
    ax[0, 2].plot(x, np.abs(y2 - y), color=cmap(2))
    ax[0, 2].set_ylabel("The Error")
    ax[0, 2].set_title("c) Cubic Spline")

    # Rational Function with Inv
    ax[1, 0].plot(x, np.abs(y3 - y), color=cmap(3))
    ax[1, 0].set_ylabel("The Error")
    ax[1, 0].set_title("d) Rational Function with Inv")

    # Rational Function with PInv
    ax[1, 1].plot(x, np.abs(y4 - y), color=cmap(4))
    ax[1, 1].set_ylabel("The Error")
    ax[1, 1].set_title("e) Rational Function with PInv")

    # Write some text on the bottom right which explains the figures.
    ax[1, 2].text(-0.1, 0.50, text, color='black', fontsize=12, wrap=True, va='center')
    ax[1, 2].axis('off')
    plt.show()

    # Return p3, q3, p4, q4 if debugging is necessary
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

# This below is to obtain the parameters for the text
# Interpolate with a Rational function (using inv and pinv) 4deg, n=4, m=6
p3, q3, y3 = interpolate(x_iq, sample_x_iq, inverseQuadratic(sample_x_iq), 3, [10, 4, 6])
p4, q4, y4 = interpolate(x_iq, sample_x_iq, inverseQuadratic(sample_x_iq), 4, [10, 4, 6])

# The figure caption
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

# End.
