import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.fft import dst, idst

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


#End time of the simulation in seconds
time_finish = 2

#Number of time steps included in the simulation
time_steps = 10

#array of time values
time = np.linspace(0, time_finish, time_steps)

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
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
a = 1.0  # length of the membrane in x-direction (m)
b = 1.0  # length of the membrane in y-direction (m)
T = 1.0  # tension (N/m)
mu = 0.01  # area mass density (kg/m^2)
c = 0.1  # damping coefficient (Pa/(m/s))
modes = 10  # number of modes to consider
t_max = 10.0  # simulation time (s)
dt = 0.01  # time step (s)
p0 = 1.0  # pressure amplitude

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

# time array
t = np.arange(0, t_max, dt)
impulse_times = [0,1,2,3,4]

# array to store displacement
w_total = np.zeros((len(t), 100, 100))

# Spatial grid
x = np.linspace(0, a, 100)
y = np.linspace(0, b, 100)
X, Y = np.meshgrid(x, y) #check later, mgrid instead?

# looping through timesteps
for ti in range(len(t)):
    w = np.zeros_like(X)
    for m in range(1, modes + 1):
        for n in range(1, modes + 1):
            k = eigvals(m, n, a, b)
            omega0_mn = np.sqrt(T * k**2 / mu)
            alpha = c / (2 * mu)
            omega_star = np.sqrt(omega0_mn**2 - alpha**2)
            
            # free oscillation solution
            A = wmn[m-1, n-1]
            B = (wmn_dot[m-1, n-1] + alpha * A) / omega_star
            wmn[m-1, n-1] = np.exp(-alpha * t[ti]) * (A * np.cos(omega_star * t[ti]) + B * np.sin(omega_star * t[ti]))
            
            # impulse handling (for example impulse at ti=1s)
            if t[ti] in impulse_times:
                wmn_dot[m-1, n-1] += p0 * p_smn(m, n, 0, a, 0, b, a, b) / mu
            
            w += wmn[m-1, n-1] * phi_mn(X, Y, m, n, a, b)
    
    w_total[ti] = w

# plotting the results  #the plots take a bit to run
plt.figure(figsize=(10, 6))
for ti in range(0, len(t), int(len(t)/10)):
    plt.contourf(X, Y, w_total[ti], levels=20, cmap='viridis') #here
    plt.colorbar()
    plt.title(f'Time = {t[ti]:.2f} s')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()