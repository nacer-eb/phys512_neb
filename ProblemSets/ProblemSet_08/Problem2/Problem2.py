import numpy as np
import matplotlib.pyplot as plt

# Use this to obtain green's function
def get_green(verbose=False, N = 128):
    """
    Obtain Green's function in 2D

    :param verboxe: Whether or not to print various infos
    :param N: System dimensions (system is set to be NxN)

    :return: The green's function centered at (N//2,N//2).
    """
    # Instantiate arrays
    V = np.zeros((N, N))
    rho = np.zeros((N, N))

    # initial conditions
    q_position = N//2
    V[q_position, q_position] = 1.0
    rho[q_position, q_position] = 1.0

    # Create mask for initial conditions (and save init. conditions)
    V_0 = V.copy()
    rho_0 = rho.copy()
    mask = rho_0 > 0

    max_loop = 10000
    # Compute V with the relaxation method (average)
    for t in range(max_loop):
        V[1:-1, 1:-1] = rho[1:-1, 1:-1] + (V[0:-2, 1:-1]+V[2:, 1:-1]+V[1:-1, 0:-2]+V[1:-1, 2:])/4.0

        # Symmetrical B.C. --  as if this grid was a 1/4 of the real grid.
        V[0, 1:-1] = rho[0, 1:-1] + (2*V[1, 1:-1]+V[0, 0:-2]+V[0, 2:])/4.0
        V[-1, 1:-1] = rho[-1, 1:-1] + (2*V[-2, 1:-1]+V[-1, 0:-2]+V[-1, 2:])/4.0
        V[1:-1, 0] = rho[1:-1, 0] + (2*V[1:-1, 1]+V[0:-2, 0]+V[2:, 0])/4.0
        V[1:-1, -1] = rho[1:-1,-1] + (2*V[1:-1, -2]+V[0:-2, -1]+V[2:, -1])/4.0
        
        V[0, 0] = rho[0, 0] + (V[1, 0] + V[0, 1])/2.0
        V[0, -1] = rho[0, -1] + (V[1, -1] + V[0, -2])/2.0
        V[-1, 0] = rho[-1, 0] + (V[-2, 0] + V[-1, 1])/2.0   
        V[-1, -1] = rho[-1, -1] + (V[-2, -1] + V[-1, -2])/2.0


    # rescaling
    V += 1 - V[q_position, q_position]

    if verbose:
        print("V[1,0]=", V[q_position+1, q_position])
        print("V[2,0]=", V[q_position+2, q_position])
        print("V[5,0]=", V[q_position+5, q_position])
        
    return V, rho


def conjugate_gradient_fit(V_target, A, max_iter, tol, N, verbose=False):
    """
    Calculates the particle density to obtain V_target, using the conjugate
    gradient method
    
    :param V_target: The target potential
    :param A: The matrix in Ax=b, in our case A is the Green's function written
              to appropritely apply a convolution
    :param max_iter: Maximum number of iterations of the conj. grad. method
    :param tol: The error tolerance in the resulting potential 
    :param N: Dimensions of the system (N is a number, the system is NxN)
    :param verboxe: Whether or not to print the number of iterations and other infos

    :return: The particle density x.
    """
    # Predifined A, V_target
    
    # Mask/Space where we want to 'fit' using the conj. grad. method.
    fit_space_mask = (V_target != 0)

    # Setup initial conditions and start the method
    x_0 = np.zeros((N*N))
    r_0 = V_target - x_0@A
    r_0[np.logical_not(fit_space_mask)] = 0
    p_0 = r_0

    x_prev = x_0.copy() # Copy to protect initial conditions - unnecessary but low-cost
    r_prev = r_0.copy()
    p_prev = p_0.copy()

    for k in range(max_iter):
        p_a = p_prev@A
        alpha = np.dot(r_prev, r_prev)/np.dot(p_prev, p_a)
        x = x_prev + alpha*p_prev
        r = r_prev - alpha*p_a
       
        beta = np.dot(r, r)/np.dot(r_prev, r_prev)
        p = r + beta*p_prev

        # Prepare next loop - using copy to be safe
        x_prev = x.copy()
        r_prev = r.copy()
        p_prev = p.copy()

        # Make sure we only care about 'fitting' the perimeter of the box
        #x_prev[np.logical_not(fit_space_mask)] = 0 # Unnecessary
        r_prev[np.logical_not(fit_space_mask)] = 0
        p_prev[np.logical_not(fit_space_mask)] = 0

        # check if r converged to tol
        if np.max(np.abs(r_prev)) < tol:
            break
    
        if k%20 == 0 and verbose:
            print(k)

    return x

# Prepare green's function for convolution
N = 150
V_green, rho_green = get_green(True, N)

# Plot the Green's function
fig = plt.figure(figsize=(12, 10))
plt.imshow(V_green, cmap="bwr")
plt.colorbar()
plt.tight_layout()
plt.title("Centered Green's Function")
plt.savefig("Green_Function_Plot.png")
plt.close()
plt.cla()
plt.clf()

# Shift the potential so our particle is at [0, 0]
V_zeroed = np.roll(V_green, (N//2, N//2), axis=(0, 1))

# Create Green's function in a convolution appropriate manner
g = np.zeros((N*N, N*N), dtype="float32")
count = 0
for i in range(0, N):
    for j in range(0, N):
        g[count] = np.roll(V_zeroed, (i, j), axis=(0, 1)).ravel()
        count += 1

# Setup the initial conditions
# Setup the box size
box_halfsize = 10

V_target = np.zeros((N, N))
V_target[N//2-box_halfsize: N//2+box_halfsize:, N//2+box_halfsize] = 1
V_target[N//2-box_halfsize: N//2+box_halfsize:, N//2-box_halfsize] = 1
V_target[N//2+box_halfsize, N//2-box_halfsize: N//2+box_halfsize+1] = 1 #Make sure the bottom right corner is included
V_target[N//2-box_halfsize, N//2-box_halfsize: N//2+box_halfsize+1] = 1 #Make sure the bottom right corner is included

V_target = V_target.ravel() # Flattern

###### Apply conjugate gradient method

x = conjugate_gradient_fit(V_target, g, 20, 0.1, N, verbose=False)

######

# Plotting 2d density
fig = plt.figure(figsize=(12, 10))
plt.imshow(np.reshape(x, (N, N)), cmap="bwr")
nplt.colorbar()
plt.title("2D Particle Density")
plt.tight_layout()
plt.savefig("2d_density.png")
plt.cla()
plt.clf()
plt.close()

# Plotting density along a box size
fig = plt.figure(figsize=(12, 10))
plt.plot(np.reshape(x, (N, N))[N//2-box_halfsize, :], ".-")
plt.title("Particle Density Along Box Side")
plt.tight_layout()
plt.savefig("density_alongside.png")
plt.cla()
plt.clf()
plt.close()


# Plotting the potential
V = np.reshape(x@g, (N, N))

fig = plt.figure(figsize=(12, 10))
plt.imshow(V, cmap="bwr")
plt.colorbar()
plt.title("The potential, V(x, y)")
plt.tight_layout()
plt.savefig("2d_potential.png")
plt.cla()
plt.clf()
plt.close()

# Plotting a slice of the potential (at the center)
plt.plot(V[N//2, :], ".-")
plt.title("A 1d slice of the potential V(N/2, y)")
plt.tight_layout()
plt.savefig("1d_V_slice.png")
plt.cla()
plt.clf()
plt.close()


# Calculating&Plotting the electric fields
# We have to write E_y this way because matplotlib defines (0, 0) in the top left corner rather than the bottom left. (Hence y axis is flipped)
E_y = -1*(V[:-2, 1:-1]-V[2:, 1:-1])/2
E_x = -1*(V[1:-1, 2:]-V[1:-1, :-2])/2

# Setting up the meshgrid for the quiver plot
x = np.arange(1, N-1, 1)
x, y= np.meshgrid(x, x)

# d is used to control arrow density in the plot (with the scale param)
d = 3
fig = plt.figure(figsize=(12, 10))
plt.quiver(x[::d, ::d], y[::d, ::d], E_x[::d, ::d], E_y[::d, ::d], np.sqrt(E_x[::d, ::d]**2 + E_y[::d, ::d]**2),
           units="xy", scale=0.005, cmap="hot")
plt.colorbar()
plt.imshow(V, cmap="bwr")
plt.colorbar()
plt.title("V(x, y) and E(x, y)")
plt.tight_layout()
plt.savefig("2d_e_field.png")
plt.cla()
plt.clf()
plt.close()

# Plotting the x-component of the electric field
plt.imshow(E_x, cmap="bwr")
plt.colorbar()
plt.title("The electric field, E_x(x, y)")
plt.tight_layout()
plt.savefig("2d_e_x_field.png")
plt.cla()
plt.clf()
plt.close()

# Plotting the y-component of the electric field
plt.imshow(E_y, cmap="bwr")
plt.colorbar()
plt.title("The electric field, E_y(x, y)")
plt.tight_layout()
plt.savefig("2d_e_y_field.png")
plt.cla()
plt.clf()
plt.close()











