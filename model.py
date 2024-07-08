import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.fft import dst, idst

class membrane_response:
    def __init__(self, impulse_times, dt=1e-6, t_max=1e-5, a=0.005, b=0.005, modes=10, h=5e-8, rho=2850, sigma=250e6, alpha=0.1, eta=0):
        self.impulse_times = impulse_times
        self.dt = dt
        self.t_max = t_max
        self.a = a
        self.b = b
        self.h = h # something that is still confusing to me is that we are treating this and plotting it like a 2D membrane, but we also give it a height so as to define tension as such?
        self.rho = rho # mass density (kg/m^3)
        self.sigma = sigma # in plane pressure (MPa)
        self.tension = self.h * self.sigma # tension (N/m)
        self.alpha = alpha  # example value
        self.eta = eta  # damping coefficient (Pa/(m/s))
        self.mu = self.rho * self.h  # area mass density (kg/m^2)
        self.num_modes = modes
        self.alpha = self.eta / (2 * self.mu)
        self.p0 = 1.0  # pressure amplitude, should i adjust this later?
        self.x = np.linspace(0, a, 100)
        self.y = np.linspace(0, b, 100)
        self.X, self.Y = np.meshgrid(self.x, self.y) #check later, mgrid instead?


    # eigenfns
    def phi_mn(self, m, n):
        return (2 / np.sqrt(self.a * self.b)) * np.sin(m * np.pi * self.x / self.a) * np.sin(n * np.pi * self.y / self.b)

    # eigenvalues
    def eigvals(self, m, n, a, b):
        return np.sqrt((m * np.pi / a)**2 + (n * np.pi / b)**2)

    #  pressure spatial mode coefficients
    def p_smn(self, m, n, x0, x1, y0, y1, a, b):
        return (a / (m * np.pi)) * (np.cos(m * np.pi * x0 / a) - np.cos(m * np.pi * x1 / a)) * \
            (b / (n * np.pi)) * (np.cos(n * np.pi * y0 / b) - np.cos(n * np.pi * y1 / b))


    # looping through timesteps
    
    def calculate_response(self):
        t = np.arange(0, self.t_max, self.dt)
        w_total = np.zeros((len(t), self.X.shape[0], self.X.shape[1]))
        w_mn = np.zeros((len(t), self.num_modes, self.num_modes))
        last_impulse_index = 0
        A = 1.0 
        B = 1.0

        for t_index in range(len(t)):
            #print(t_index)
            current_time = t[t_index]
            #print(current_time) #debugging: this is making sure it enters the loop
            print(last_impulse_index) #debugging: this is seeing if it even find the next impulses

            #w = np.zeros_like(self.X)
            for m in range(1, self.num_modes + 1):
                for n in range(1, self.num_modes + 1):
                    k = self.eigvals(m, n, self.a, self.b)
                    pS_mn = self.p_smn(m, n, 0, self.a, 0, self.b, self.a, self.b)
                    omega0_mn = np.sqrt(self.tension * k**2 / self.mu)
                    #print(self.alpha) 
                    omega_star = np.sqrt(omega0_mn**2 - self.alpha**2)
                    #print(omega_star/2*np.pi)

                # checking if current time against next impulse
                    #debugging that last_imp_index isn't being picked up: first if condition, when commented out, still outputs 0
                    if (last_impulse_index != 0 & last_impulse_index + 1 < len(self.impulse_times)) and \
                        current_time >= self.impulse_times[last_impulse_index + 1]:
                        print("hello i am the debugger :D")

                        # shift time reference
                        t_shifted = self.impulse_times[last_impulse_index + 1] - self.impulse_times[last_impulse_index]

                        # calculate initial conditions for wmn and wmn_dot at the time of the current impulse
                        wmn_init = np.exp(-self.alpha * t_shifted) * (A * np.cos(omega_star * t_shifted) + B * np.sin(omega_star * t_shifted))
                        wmn_dot_init_minus = np.exp(-self.alpha * t_shifted) * (B * omega_star * np.cos(omega_star * t_shifted) - \
                                                                                A * omega_star * np.sin(omega_star * t_shifted) - \
                                                                                A * self.alpha * np.cos(omega_star * t_shifted) - \
                                                                                B * self.alpha * np.sin(omega_star * t_shifted))

                        # applying the velocity jump condition
                        wmn_dot_init = wmn_dot_init_minus + self.p0 * pS_mn / self.mu

                        # updating A and B, impulse index
                        A = wmn_init
                        B = (wmn_dot_init + self.alpha * wmn_init) / omega_star
                        last_impulse_index += 1

                    # calculating wmn at the current time with reference to the last impulse
                    t_shifted = current_time - self.impulse_times[last_impulse_index]
                    w_mn[t_index, m - 1, n - 1] = np.exp(-self.alpha * t_shifted) * (A * np.cos(omega_star * t_shifted) + B * np.sin(omega_star * t_shifted))
                    w_total[t_index] = w_mn[t_index, m - 1, n - 1] * self.phi_mn(m, n)

        return t, w_total, w_mn
    """
    # initial conditions
    wmn = np.zeros((modes, modes))
    wmn_dot = np.zeros((modes, modes))
    #wmn[0, 0] = 1  # testing testing!! just the phi_1_1 mode

    # time array
    t = np.arange(0, t_max, dt)
    impulse_times = [0]
    # impulse_times = [0.2, 0.3, 1, 1.1]
    # impulse_times = [1, 3, 4, 7, 9] commented out for debugging

    # array to store displacement
    w_total = np.zeros((len(t), 100, 100))

    # spatial grid
    x = np.linspace(0, a, 100)
    y = np.linspace(0, b, 100)
    X, Y = np.meshgrid(x, y) #check later, mgrid instead?


    ti = 0
    while ti in range(len(t)):
        w = np.zeros_like(X)
        for m in range(1, modes + 1):
            for n in range(1, modes + 1):
                k = eigvals(m, n, a, b)
                omega0_mn = np.sqrt(T * k**2 / mu)
                alpha = eta / (2 * mu)
                #print(alpha) 
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
        """
    
    
    def plot_displacement(self, w_total, t):
        plt.figure(figsize=(10, 6))
        for ti in range(len(t)): # adjust later, have the factor that len(t) is divided by depend on time steps
            plt.contourf(self.X, self.Y, w_total[ti], cmap='magma')
            plt.colorbar()
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f'Displacement at t = {t[ti]:.2f} s')
            plt.savefig(f'displacement_{ti}.png', format="png")
            plt.show()


    # plotting the avg displacement as a function of time
    def plot_avg_displacement_vs_time(w_total, t):
        avg_displacement = np.mean(w_total, axis=(1, 2)) # this is because w_total is an array [# time steps, # x's, # y's]
        plt.plot(t, avg_displacement)
        plt.xlabel('Time')
        plt.ylabel('Average Displacement (w)')
        plt.title('Average Displacement vs. Time')
        plt.savefig(f'avg_displacement_v_time.png', format="png")
        plt.show()

    # plotting the displacement as a function of time
    def plot_displacement_vs_time(self, w, t):
        plt.figure()
        for i in range(len(self.x)):
            for j in range(len(self.y)):
                plt.plot(t, w[i, j, :], label=f'Point ({self.x[i]:.2f}, {self.y[j]:.2f})')
        plt.xlabel('Time (s)')
        plt.ylabel('Displacement')
        plt.title('Displacement vs Time')
        plt.legend()
        plt.savefig('displacement_vs_time.png', format="png")
        plt.show()


    # plotting the displacement along a given plane (e.g., y = b/2)
    def plot_cutout_along_plane(self, w_total, plane='y', value=0.5):
        plt.figure()
        if plane == 'y':
            idx = (np.abs(self.y - value)).argmin()
            for ti in range(len(t)):
                plt.plot(self.x, w_total[ti, idx, :], label=f'Time = {t[ti]:.2f}s')
            plt.xlabel('Distance along membrane (x)')
        elif plane == 'x':
            idx = (np.abs(self.x - value)).argmin()
            for ti in range(len(t)):
                plt.plot(self.y, w_total[ti, :, idx], label=f'Time = {t[ti]:.2f}s')
            plt.xlabel('Distance along membrane (y)')
        plt.ylabel('Displacement (w)')
        plt.title(f'Displacement along {plane}={value}')
        plt.legend()
        plt.savefig(f'displacement_{plane}_{value}_cutout_{ti}.png', format="png")
        plt.show()
    

    # plot individual modes over time (just w_1_1)
    def plot_individual_modes(self, w_mn, t):
        time_plot_arr = range(0, len(t), int(len(t)/10))
        plt.figure()
        plt.plot(time_plot_arr, w_mn[time_plot_arr, 0, 0], label='w_1_1') 
        plt.xlabel('Time')
        plt.ylabel('Displacement')
        plt.title('Individual Modes Over Time')
        plt.legend()
        plt.savefig('individual_modes_over_time.png', format="png")
        plt.show()


# calling membrane response
impulse_times = [0,1,2,3]
deflection = membrane_response(impulse_times)
t, w_total, w_mn = deflection.calculate_response()

print(t) #debugging

# plotting the results  #the plots take a bit to run
# this plot should be the main membrane response
#deflection.plot_displacement(w_total, t)

# calls the plotting fn for displacement v time
#deflection.plot_displacement_vs_time(w_total, t)

# calls the plotting fn for avg displacement v time
#deflection.plot_avg_displacement_vs_time(w_total, t)

# calls the plotting fn for cutout of the response
#cutout_line = (deflection.b)/2
#deflection.plot_cutout_along_plane(w_total, plane='y', value=cutout_line)

# uncomment to check individual modes of the displacement response
deflection.plot_individual_modes(w_mn, t)

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

"""
# parameters (most are inputted as default parameters for an instant of the membrane response class)
a = 0.005  # length of the membrane in x-direction (m)
b = 0.005  # length of the membrane in y-direction (m)
h = 5e-8 # membrane thickness
rho = 2850 # mass density (kg/m^3)
sigma = 250e6 # in plane pressure (MPa)
T = h * sigma  # tension (N/m)
mu = rho * h  # area mass density (kg/m^2)
eta = 0  # damping coefficient (Pa/(m/s))
modes = 10 # number of modes to consider    # prob try a lot more modes
t_max = 1e-5  # simulation time (s)
dt = 1e-6  # time step (s)
p0 = 1.0  # pressure amplitude # should i adjust this later?
"""