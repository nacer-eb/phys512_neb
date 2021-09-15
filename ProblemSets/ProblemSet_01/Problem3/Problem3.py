import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("lakeshore.txt", skiprows=3).T

data_even = data.T[0::2].T
data_odd = data.T[1::2].T

# The concatenation is added to define a value (of zero) for dT/dV at the last point
# which then will include it in the interpolation
dT = data[0][0:]-np.concatenate((data[0][1:], [data[0][-1]]))
dV = data[1][0:]-np.concatenate((data[1][1:], [0]))

def getClosestIndex(v_in, v_data):
    # Changes a constant with an array with size 1, and does nothing to arrays
    v_arr = np.atleast_1d(np.multiply(v_in, 1.0))

    # Make sure you are within range (Note the voltage is decreasing)
    assert (v_arr >= v_data[-1]).all()
    assert (v_arr <= v_data[0]).all()

    # Look for the closest indices of each v_arr[i]
    closest_indices = np.zeros(len(v_arr), dtype=int)
    for i in range(len(v_arr)):
        closest_indices[i] = np.where(v_arr[i] <= v_data)[0][-1]
    
    return closest_indices


def lakeshore(V, data):
    # For readability purposes
    data_t = data[0]
    data_v = data[1]

    # Get the closest indices
    closest_indices = getClosestIndex(V, data_v)

    # Then get the closest values of V,T (V_0, T_0)
    closest_values_v = np.take(data_v, closest_indices)
    closest_values_t = np.take(data_t, closest_indices)

    # Calculate the slope at each V, using dT, dV and taking the closest slope (by index)
    slope = np.take(dT/dV, closest_indices)

    # Approximating the error
    # Essentially remove the closest point and look at how well the interpolation
    # would do there and use that as the error approximation.
    # This is involving the 2nd derivative to obtain an error estimate.
    ci = closest_indices

    #Make sure we are doing our estimate of the error away from the boundary.
    # (Using our B.C.)
    ci = np.where(ci == 0, 1, ci)
    ci = np.where(ci == len(data_t)-1, len(data_t)-2, ci)

    new_slope = (np.take(data_t, ci+1) - np.take(data_t, ci-1))/(np.take(data_v, ci+1)
                                                                 - np.take(data_v, ci-1))

    # this all works out to be : interpolated_point - actual value
    delta_slope = new_slope - np.take(dT/dV, ci)
    dv = np.take(data_v, ci) - np.take(data_v, ci-1)

    # The error estimate for each point
    err_estimate = np.abs(delta_slope*dv)

    # interpolation
    interpolated_values = closest_values_t + (V - closest_values_v)*slope
    return err_estimate, interpolated_values


v = np.linspace(data[1][0], data[1][-1], 1000)
err, t = lakeshore(v, data)


plt.plot(v, t)
plt.show()

plt.plot(t, err/t, color="red")
plt.show()
