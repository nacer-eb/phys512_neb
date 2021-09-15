import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import warnings

warnings.filterwarnings('ignore')

def getSamplePoints(x, x_data, y_data, npts):
    # Changes a constant with an array with size 1, and does nothing to arrays
    x_arr = np.atleast_1d(np.multiply(x, 1.0))

    # Make sure you are within range (Note the voltage is decreasing)
    assert (x_arr >= np.min(x_data)).all()
    assert (x_arr <= np.max(x_data)).all()

    # Look for the closest indices of each v_arr[i]
    closest_indices = np.zeros(len(x_arr), dtype=int)
    for i in range(len(x_arr)):
        closest_indices[i] = np.where(x_arr[i] >= x_data)[0][-1]

    # Safeguard against being at the boundary
    closest_indices = np.where(closest_indices == 0, int(npts/2)-1, closest_indices)
    closest_indices = np.where(closest_indices >= len(x_data)-int(npts/2)-1,
                               len(x_data)-int(npts/2)-2, closest_indices)
    
    # Create the index at which you want your sample points, assuming symmetric sampling.
    index_range = [0]*npts
    for i in range(0, int(npts)):
        index_range[i] = closest_indices+(i-int(npts/2)+1)

    # Take each of the sample points and put them into an array. [x-index][sample point index]
    x_sample = np.take(x_data, index_range).T
    y_sample = np.take(y_data, index_range).T
    
    return x_sample, y_sample


def poly_interpolate(x, x_data, y_data, deg):
    # Turn any constant into a 1D array
    x_arr = np.atleast_1d(np.multiply(x, 1.0))

    y = np.zeros(len(x_arr))
    x_sample, y_sample = getSamplePoints(x, x_data, y_data, deg) # By default nptns = deg
    for i in range(len(x_arr)):
        p = np.polyfit(x_sample[i], y_sample[i], deg)
        y[i] = np.polyval(p, x_arr[i])
        
    return y

def cubicspln_interpolate(x, x_data, y_data, npts):
    # Turn any constant into a 1D array
    x_arr = np.atleast_1d(np.multiply(x, 1.0))

    y = np.zeros(len(x_arr))
    x_sample, y_sample = getSamplePoints(x, x_data, y_data, npts) # Get sample points
    for i in range(len(x_arr)):       
        cs = CubicSpline(x_sample[i], y_sample[i])
        y[i] = cs(x_arr[i])
        
    return y


x = np.linspace(0,3, 100)
smaller_x = np.linspace(0, 3, 20)
y = cubicspln_interpolate(x, smaller_x, np.sin(smaller_x), 4)

plt.plot(x, y-np.sin(x))
# plt.scatter(smaller_x, np.sin(smaller_x))
plt.show()

x = np.linspace(0,3, 100)
smaller_x = np.linspace(0, 3, 20)
y = poly_interpolate(x, smaller_x, np.sin(smaller_x), 4)

plt.plot(x, y-np.sin(x))
# plt.scatter(smaller_x, np.sin(smaller_x))
plt.show()

