import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.fft import dst, idst


# parameters
a = 0.005  # length of the membrane in x-direction (m)
b = 0.005  # length of the membrane in y-direction (m)
h = 5e-8 # membrane thickness
rho = 2850 # mass density (kg/m^3)
sigma = 250e6 # in plane pressure (MPa)
T = h * sigma  # tension (N/m)
mu = rho * h  # area mass density (kg/m^2)
eta = 0  # damping coefficient (Pa/(m/s))
modes = 100 # number of modes to consider    # prob try a lot more modes
t_max = 3e-5  # simulation time (s)
dt = 1e-6  # time step (s)
p0 = 1.0  # pressure amplitude # should i adjust this later?


"""
# testing old parameters again for debugging purposes
a = 1.0  # length of the membrane in x-direction (m)
b = 1.0  # length of the membrane in y-direction (m)
T = 1.0  # tension (N/m)
mu = 0.01  # area mass density (kg/m^2)
eta = 0.1  # damping coefficient (Pa/(m/s))
modes = 10  # number of modes to consider
t_max = 10.0  # simulation time (s)
dt = 0.01  # time step (s)
# dt = 1 #commented out for debugging
p0 = 1.0  # pressure amplitude
"""

# eigenfns
def phi_mn(x, y, m, n, a, b):
    return (2 / np.sqrt(a * b)) * np.sin(m * np.pi * x / a) * np.sin(n * np.pi * y / b)

# eigenvalues
def eigvals(m, n, a, b):
    return np.sqrt((m * np.pi / a)**2 + (n * np.pi / b)**2)

#  pressure spatial mode coefficients
def p_smn(m, n, x0, x1, y0, y1, a, b):
    return (a / (m * np.pi)) * (np.cos(m * np.pi * x0 / a) - np.cos(m * np.pi * x1 / a)) * \
           (b / (n * np.pi)) * (np.cos(n * np.pi * y0 / b) - np.cos(n * np.pi * y1 / b))

# initial conditions
wmn = np.zeros((modes, modes))
wmn_dot = np.zeros((modes, modes))
#wmn[0, 0] = 1  # testing testing!! just the phi_1_1 mode

# time array
t = np.arange(0, t_max, dt)
impulse_times = [0, 1, 2, 3, 4]
# impulse_times = [0.2, 0.3, 1, 1.1]
# impulse_times = [1, 3, 4, 7, 9] commented out for debugging

# array to store displacement
w_total = np.zeros((len(t), 100, 100))

# spatial grid
x = np.linspace(0, a, 100)
y = np.linspace(0, b, 100)
X, Y = np.meshgrid(x, y) #check later, mgrid instead?

# looping through timesteps
ti = 0
while ti in range(len(t)):
    w = np.zeros_like(X)
    for m in range(1, modes + 1):
        for n in range(1, modes + 1):
            k = eigvals(m, n, a, b)
            omega0_mn = np.sqrt(T * k**2 / mu)
            alpha = eta / (2 * mu)
            #print(alpha) #printing alpha to terminal to debug
            omega_star = np.sqrt(omega0_mn**2 - alpha**2)
            #print(omega_star/2*np.pi)
            
            # free oscillation solution
            A = wmn[m-1, n-1]
            B = (wmn_dot[m-1, n-1] + alpha * A) / omega_star
            wmn[m-1, n-1] = np.exp(-alpha * t[ti]) * (A * np.cos(omega_star * t[ti]) + B * np.sin(omega_star * t[ti]))
            
            # impulse handling (for example impulse at ti=1s)
            if t[ti] in impulse_times:
                wmn_dot[m-1, n-1] += p0 * p_smn(m, n, 0, a, 0, b, a, b) / mu
            
            w += wmn[m-1, n-1] * phi_mn(X, Y, m, n, a, b)
    
    w_total[ti] = w
    ti += 1

# plotting the results  #the plots take a bit to run
# this plot should be the main membrane response
plt.figure(figsize=(10, 6))
for ti in range(0, len(t), int(len(t)/10)): # adjust later, have the factor that len(t) is divided by depend on time steps
    plt.contourf(X, Y, w_total[ti], levels=20, cmap='viridis') #here
    plt.colorbar()
    plt.title(f'Membrane Displacement Response at Time = {t[ti]:.2f} s')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


# plotting the displacement as a function of time
def plot_displacement_vs_time(w_total, t, x, y):
    avg_displacement = np.mean(w_total, axis=(1, 2)) # this is because w_total is an array [# time steps, # x's, # y's]
    plt.plot(t, avg_displacement)
    plt.xlabel('Time')
    plt.ylabel('Average Displacement (w)')
    plt.title('Average Displacement vs. Time')
    plt.savefig(f'avg_displacement_v_time.png', format="png")
    plt.show()

# plotting the displacement along a given plane (e.g., y = b/2)
def plot_cutout_along_plane(w_total, x, y, plane='y', value=0.5):
    plt.figure()
    if plane == 'y':
        idx = (np.abs(y - value)).argmin()
        for ti in range(len(t)):
            plt.plot(x, w_total[ti, idx, :], label=f'Time = {t[ti]:.2f}s')
        plt.xlabel('Distance along membrane (x)')
    elif plane == 'x':
        idx = (np.abs(x - value)).argmin()
        for ti in range(len(t)):
            plt.plot(y, w_total[ti, :, idx], label=f'Time = {t[ti]:.2f}s')
        plt.xlabel('Distance along membrane (y)')
    plt.ylabel('Displacement (w)')
    plt.title(f'Displacement along {plane}={value}')
    plt.legend()
    plt.savefig(f'displacement_{plane}_{value}_cutout_{ti}.png', format="png")
    plt.show()

"""
# plot individual modes over time (just w_1_1, w_1_2)
def plot_individual_modes(wmn, t, output_dir='membrane_plots'):
    plt.figure()
    plt.plot(t, wmn[:, 0, 0], label='w_1_1')
    plt.plot(t, wmn[:, 0, 1], label='w_1_2')
    plt.xlabel('Time')
    plt.ylabel('Displacement')
    plt.title('Individual Modes Over Time')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'individual_modes_over_time.png'))
    plt.close()
"""


# calls the plotting fn for displacement v time
plot_displacement_vs_time(w_total, t, x, y)

# calls the plotting fn for cutout of the response
plot_cutout_along_plane(w_total, x, y, plane='y', value=b/2)

# uncomment to check individual modes of the displacement response
# plot_individual_modes(wmn, t,)

"""
# ignore this commented part this is just me referrring to will's code
# dimensions of membrane
# Should be the same, this is a square membrane
width = 0.005
height = 0.005
# associated membrane quantities
sigma = 250e6 
eta = 1
rho = 2850
h = 100e-9
# number of sections of membrane in the x (nx) and y (ny) directions
nx = 21
ny = 21

'''
Here are some ways of creating a time dependent forcing function. The 'frequency' variable is useful for simulating
Gaussian functions or eigenfunction pressure functions at a given sinusoidal time depedence. Some examples are commented
out since only one ff function should be used.

frequency = 1 #ask if this i should have based on webb model
# ff = lambda x, y, t: np.sin(2 * np.pi * 200 * x) * np.sin(2 * np.pi * 100 * y) + np.sin(2 * np.pi * 100 * x) * np.sin(
#     2 * np.pi * 100 * y) + np.sin(2 * np.pi * 300 * x) * np.sin(2 * np.pi * 200 * y) * np.sin(2 * np.pi * frequency * t)

# ff = lambda x, y, t: ( np.sin(2 * np.pi * 100 * x) * np.sin(
#     2 * np.pi * 100 * y)) * np.sin(2 * np.pi * frequency * t)


ff = lambda x, y, t: (np.sin(2 * np.pi * 100 * x) * np.sin(
    2 * np.pi * 100 * y)) * np.sin(2 * np.pi * frequency * t)
'''

frequency = 1

use will's ffs in imesolver.py for verifications
"""