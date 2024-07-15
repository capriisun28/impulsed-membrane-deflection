import numpy as np
import matplotlib.pyplot as plt
# from scipy.integrate import solve_ivp 
# from scipy.fft import dst, idst

class membrane_response:
    def __init__(self, impulse_times, dt=1e-8, t_max=1e-5, a=0.008, b=0.005, modes=10, h=5e-8, rho=2850, sigma=250e6, eta=0.5):
        self.impulse_times = impulse_times
        self.dt = dt
        self.t_max = t_max
        self.a = a
        self.b = b
        self.h = h
        self.rho = rho # mass density (kg/m^3)
        self.sigma = sigma # in plane pressure (MPa)
        self.tension = self.h * self.sigma # tension (N/m)
        self.eta = eta  # damping coefficient (Pa/(m/s))
        self.mu = self.rho * self.h  # area mass density (kg/m^2)
        self.num_modes = modes
        self.alpha = self.eta / (2 * self.mu)
        self.p0 = 1.0  # pressure amplitude, can adjust later if wanting to specify strength of force
        self.x = np.linspace(0, a, 100)
        self.y = np.linspace(0, b, 100)
        self.X, self.Y = np.meshgrid(self.x, self.y) 

    # eigenfns
    def phi_mn(self, m, n):
        return (2 / np.sqrt(self.a * self.b)) * np.sin(m * np.pi * self.X / self.a) * np.sin(n * np.pi * self.Y / self.b)

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
        w_mn_dot = np.zeros((len(t), self.num_modes, self.num_modes))
        w_mn_dot_minus = np.zeros((len(t), self.num_modes, self.num_modes))
        wmn_init = 0
        wmn_dot_init = 0

        for m in range(1, self.num_modes + 1):
            for n in range(1, self.num_modes + 1):
                print("-----------------------------------------------------------------------")

                k = self.eigvals(m, n, self.a, self.b)
                pS_mn = self.p_smn(m, n, (self.a / 3), (2 * self.a / 3), (self.b/3), (2 * self.b / 3), self.a, self.b)
                omega0_mn = np.sqrt(self.tension * k**2 / self.mu)
                omega_star = np.sqrt(omega0_mn**2 - self.alpha**2)
                #print(omega_star/2*np.pi)
                enter = True
                A = 0.0 
                B = 0.0
                curr_impulse_index = 0

                for t_index in range(len(t)):
                    #print(t_index)
                    current_time = t[t_index]
            
                    # checking if current time against next impulse
                    if (enter) and (curr_impulse_index == 0 and current_time >= self.impulse_times[0]):
                        enter = False
                        wmn_dot_init = self.p0 * pS_mn / self.mu
                        B = wmn_dot_init / omega_star
                        print("knock knock, A: ", A, ", B: ", B, ", wmn_dot_init: ", wmn_dot_init)
                        
                    #debugging that last_imp_index isn't being picked up: first if condition, when commented out, still outputs 
                    if (curr_impulse_index + 1 < len(self.impulse_times) and \
                        (current_time >= self.impulse_times[curr_impulse_index + 1])):
                        
                        # shift time reference
                        t_shifted = self.impulse_times[curr_impulse_index + 1] - self.impulse_times[curr_impulse_index]
                        
                        # calculate initial conditions for wmn and wmn_dot at the time of the current impulse
                        wmn_init = np.exp(-self.alpha * t_shifted) * (A * np.cos(omega_star * t_shifted) + B * np.sin(omega_star * t_shifted))
                        wmn_dot_init_minus = np.exp(-self.alpha * t_shifted) * (B * omega_star * np.cos(omega_star * t_shifted) - \
                                                                                A * omega_star * np.sin(omega_star * t_shifted) - \
                                                                                A * self.alpha * np.cos(omega_star * t_shifted) - \
                                                                                B * self.alpha * np.sin(omega_star * t_shifted))
                        
                        print("hello i am the debugger :D", "| t_shifted in if: ", t_shifted, "| wmn init: ", wmn_init, "| wmn dot minus: ", wmn_dot_init_minus)
                        # applying the velocity jump condition
                        wmn_dot_init = wmn_dot_init_minus + self.p0 * pS_mn / self.mu

                        # updating A and B, impulse index
                        A = wmn_init
                        B = (wmn_dot_init + self.alpha * wmn_init) / omega_star
                        curr_impulse_index += 1

                        #debugging w_mn plots not changing even when w_mn changes:
                        # calculating wmn at the current time with reference to the last impulse
                    if current_time < self.impulse_times[0]:
                        t_shifted = current_time
                    else:
                        t_shifted = current_time - self.impulse_times[curr_impulse_index]
                    
                    
                    w_mn[t_index, m - 1, n - 1] = np.exp(-self.alpha * t_shifted) * (A * np.cos(omega_star * t_shifted) + B * np.sin(omega_star * t_shifted))
                    #these two are for plotting
                    w_mn_dot_minus[t_index, m - 1, n - 1] = wmn_dot_init
                    w_mn_dot[t_index, m - 1, n - 1] = np.exp(-self.alpha * t_shifted) * (B * omega_star * np.cos(omega_star * t_shifted) - \
                                                                                A * omega_star * np.sin(omega_star * t_shifted) - \
                                                                                A * self.alpha * np.cos(omega_star * t_shifted) - \
                                                                                B * self.alpha * np.sin(omega_star * t_shifted))
                    w_total[t_index] += w_mn[t_index, m - 1, n - 1] * self.phi_mn(m, n)
                    print("t_shifted: ", t_shifted, "|| curr t: ", current_time, "|| curr imp t: ", self.impulse_times[curr_impulse_index], "|| curr imp ind: ", curr_impulse_index, " || m: ", m, "|| n :", n, "|| A: ", A, "|| B: ", B)

        return t, w_total, w_mn, w_mn_dot_minus, w_mn_dot

    
    def plot_displacement(self, w_total, t):
        plt.figure(figsize=(10, 6))
        for ti in range(0, len(t), int(len(t)/100)): # adjust later, have the factor that len(t) is divided by depend on time steps
            plt.contourf(self.x, self.y, w_total[ti], levels=20, cmap='magma')
            plt.colorbar(label = 'Displacement')
            plt.title(f'Membrane Displacement Response at Time = {t[ti]:.2f} s')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.savefig(f'displacement_{ti}.png', format="png")
            plt.show()


    # plotting the avg displacement as a function of time
    def plot_avg_displacement_vs_time(self, w_total, t):
        avg_displacement = np.mean(w_total, axis=(1, 2))
        plt.plot(t, avg_displacement)
        plt.xlabel('Time')
        plt.ylabel('Average Displacement (w)')
        plt.title('Average Displacement vs. Time')
        plt.savefig(f'avg_displacement_v_time.png', format="png")
        plt.show()

    # plotting the displacement as a function of time #this runs really slow rn
    def plot_displacement_vs_time(self, w_total, t):
        plt.figure()
        for i in range(0, len(self.x)):
            for j in range(0, len(self.y)):
                plt.plot(t, w_total[:, i, j], label=f'Point ({self.x[i]:.2f}, {self.y[j]:.2f})')
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
                plt.plot(self.x, w_total[ti, idx, :], label=f'Time = {t[ti]:.2f}s') # fix label
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
    def plot_individual_modes(self, w_mn, t, m, n):
        time_plot_arr = range(0, len(t))
        plt.figure(figsize=(11, 7))
        plt.plot(t, w_mn[time_plot_arr, m - 1, n - 1], label=f'w_{m}_{n}') #m,n and n,m plot the same
        plt.xticks(np.arange(0, max(t)+self.dt, self.dt/2)) # sorry this is kinda cursed if your dt is tiny .. ill fix it i promise
        plt.xlabel('Time')
        plt.ylabel('Displacement')
        plt.title('Individual Modes Over Time')
        plt.legend()
        plt.savefig('individual_modes_over_time.png', format="png")
        plt.show()

    def plot_velocity_imparted_over_time(self, w_mn_dot_minus, t):
        time_plot_arr = range(0, len(t))
        plt.figure(figsize=(11, 7))
        plt.scatter(t, w_mn_dot_minus[time_plot_arr, 7, 5], label='velocity') 
        plt.xticks(np.arange(0, max(t)+self.dt, self.dt/2))
        #plt.yticks(np.arange(0,1e-30, 1e-32))
        plt.xlabel('Time')
        plt.ylabel('Velocity')
        plt.title('Velocity After Imparted Impulse')
        plt.legend()
        plt.savefig('velocity_over_time.png', format="png")
        plt.show()

    def plot_velocity_over_time(self, w_mn_dot, t):
        time_plot_arr = range(0, len(t))
        plt.figure(figsize=(11, 7))
        plt.plot(t, w_mn_dot[time_plot_arr, 7, 5], label='velocity') 
        plt.xticks(np.arange(0, max(t)+self.dt, self.dt/2))
        #plt.yticks(np.arange(0,1e-8, 1e-11))
        plt.xlabel('Time')
        plt.ylabel('Velocity')
        plt.title('Velocity vs Time')
        plt.legend()
        plt.savefig('velocity_over_time.png', format="png")
        plt.show()


# calling membrane response
print("hi!") #easy flag to scroll to the start

# test case 1 (ideal parameters, velocity vs time graph looks weird, continuity on displacement graph also looks odd)
impulse_times = [0, 1e-6, 2e-6, 3e-6]
deflection = membrane_response(impulse_times, dt=1e-7)

# test case 2 (woah :OOO dt = 0.01 looks real cool. 
#             but dt = 0.1 is easier to see the behavior of, and that's what I've been using to debug)
#impulse_times = [0.6, 1, 2]
#deflection = membrane_response(impulse_times, dt=0.5, t_max=2.51, h=5e-4, eta=2)
t, w_total, w_mn, w_mn_dot_minus, w_mn_dot = deflection.calculate_response()

#print(w_mn)

# plotting the results  #the plots take a bit to run
# this plot should be the main membrane response
deflection.plot_displacement(w_total, t)

# calls the plotting fn for displacement v time
# commented out for now as this runs really slowly
#deflection.plot_displacement_vs_time(w_total, t)

# calls the plotting fn for avg displacement v time
#deflection.plot_avg_displacement_vs_time(w_total, t)

# calls the plotting fn for cutout of the response
#cutout_line = (deflection.b)/2
#deflection.plot_cutout_along_plane(w_total, plane='y', value=cutout_line)

# uncomment to check individual modes of the displacement response
deflection.plot_individual_modes(w_mn, t, 9, 7)
#deflection.plot_velocity_imparted_over_time(w_mn_dot_minus, t)
#deflection.plot_velocity_over_time(w_mn_dot, t)

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