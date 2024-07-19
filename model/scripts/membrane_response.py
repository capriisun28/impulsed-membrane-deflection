import numpy as np
import math

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
        self.p0 = 3.336e-8  #value provided by Adam #N-s/(m^2) # pressure amplitude, can adjust later if wanting to specify strength of force
        self.x = np.linspace(0, a, 100)
        self.y = np.linspace(0, b, 100)
        self.X, self.Y = np.meshgrid(self.x, self.y) 

    ### eigenfns
    def phi_mn(self, m, n):
        return (2 / np.sqrt(self.a * self.b)) * np.sin(m * np.pi * self.X / self.a) * np.sin(n * np.pi * self.Y / self.b)

    ### eigenvalues
    def eigvals(self, m, n, a, b):
        return np.sqrt((m * np.pi / a)**2 + (n * np.pi / b)**2)

    ###  pressure spatial mode coefficients
    def p_smn(self, m, n, x0, x1, y0, y1, a, b):
        return (a / (m * np.pi)) * (np.cos(m * np.pi * x0 / a) - np.cos(m * np.pi * x1 / a)) * \
               (b / (n * np.pi)) * (np.cos(n * np.pi * y0 / b) - np.cos(n * np.pi * y1 / b))

    ### looping through timesteps
    def calculate_response(self):

        t = np.arange(0, self.t_max, self.dt)
        w_total = np.zeros((len(t), self.X.shape[0], self.X.shape[1]))
        w_mn = np.zeros((len(t), self.num_modes, self.num_modes))
        w_mn_dot = np.zeros((len(t), self.num_modes, self.num_modes))
        w_mn_dot_minus = np.zeros((len(t), self.num_modes, self.num_modes))
        wmn_init = 0
        wmn_dot_init = 0
        Q = np.zeros((self.num_modes + 1, self.num_modes + 1))
        omega_resonant = np.zeros((self.num_modes + 1, self.num_modes + 1))

        for m in range(1, self.num_modes + 1):
            for n in range(1, self.num_modes + 1):
                print("-----------------------------------------------------------------------")

                k = self.eigvals(m, n, self.a, self.b)

                if (self.eta**2 >= 4 * self.mu * self.tension * (k**2)):
                    print("This response is overdamped or critically damped. The model currently accounts only for the underdamped case.")
                    break

                pS_mn = self.p_smn(m, n, ((self.a / 2) - (self.a / 20)), ((self.a / 2) + (self.a / 20)), ((self.b / 2) - (self.a / 20)), ((self.b / 2) + (self.a / 20)), self.a, self.b)
                ### offcentered incident force
                #pS_mn = self.p_smn(m, n, ((self.a / 3) - (self.a / 20)), ((self.a / 3) + (self.a / 20)), ((self.b / 3) - (self.a / 20)), ((self.b / 3) + (self.a / 20)), self.a, self.b)
                omega0_mn = np.sqrt(self.tension * k**2 / self.mu)
                omega_star = np.sqrt(omega0_mn**2 - self.alpha**2)
                Q[m, n] = omega0_mn / (2 * self.alpha)
                omega_resonant[m, n] = math.sqrt((omega0_mn**2) - ((self.eta**2)/(2 * (self.mu**2))))
                #print("alpha: ", omega_star/2*np.pi, "|| omega0: ",omega0_mn, "|| omega*: ", omega_star)
                #print("Q: ", Q[m,n], "resonant frequency: ", omega_resonant[m,n])
                enter = True
                A = 0.0 
                B = 0.0
                curr_impulse_index = 0

                for t_index in range(len(t)):
                    #print(t_index)
                    current_time = t[t_index]
            
                    ### checking current time against first impulse time
                    if (enter) and (curr_impulse_index == 0 and current_time >= self.impulse_times[0]):
                        enter = False
                        wmn_dot_init = self.p0 * pS_mn / self.mu
                        B = wmn_dot_init / omega_star
                        print("knock knock, A: ", A, ", B: ", B, ", wmn_dot_init: ", wmn_dot_init)
                        
                    ### checking current time against subsequent impulse times
                    if (curr_impulse_index + 1 < len(self.impulse_times) and \
                        (current_time >= self.impulse_times[curr_impulse_index + 1])):
                        
                        ### shift time reference
                        t_shifted = self.impulse_times[curr_impulse_index + 1] - self.impulse_times[curr_impulse_index]
                        
                        ### calculate initial conditions for wmn and wmn_dot at the time of the current impulse
                        wmn_init = np.exp(-self.alpha * t_shifted) * (A * np.cos(omega_star * t_shifted) + B * np.sin(omega_star * t_shifted))
                        wmn_dot_init_minus = np.exp(-self.alpha * t_shifted) * (B * omega_star * np.cos(omega_star * t_shifted) - \
                                                                                A * omega_star * np.sin(omega_star * t_shifted) - \
                                                                                A * self.alpha * np.cos(omega_star * t_shifted) - \
                                                                                B * self.alpha * np.sin(omega_star * t_shifted))
                        ### terminal output, comment out if you don't want to clog up the terminal
                        print("hello i am the debugger :D", "| t_shifted in if: ", t_shifted, "| wmn init: ", wmn_init, "| wmn dot minus: ", wmn_dot_init_minus)
                        ### applying the velocity jump condition
                        wmn_dot_init = wmn_dot_init_minus + self.p0 * pS_mn / self.mu

                        ### updating A and B, impulse index
                        A = wmn_init
                        B = (wmn_dot_init + self.alpha * wmn_init) / omega_star
                        curr_impulse_index += 1

                    if current_time < self.impulse_times[0]:
                        t_shifted = current_time
                    else:
                        t_shifted = current_time - self.impulse_times[curr_impulse_index]
                
                    w_mn[t_index, m - 1, n - 1] = np.exp(-self.alpha * t_shifted) * (A * np.cos(omega_star * t_shifted) + B * np.sin(omega_star * t_shifted))
                    ### these two arrays are for plotting
                    w_mn_dot_minus[t_index, m - 1, n - 1] = wmn_dot_init
                    w_mn_dot[t_index, m - 1, n - 1] = np.exp(-self.alpha * t_shifted) * (B * omega_star * np.cos(omega_star * t_shifted) - \
                                                                                A * omega_star * np.sin(omega_star * t_shifted) - \
                                                                                A * self.alpha * np.cos(omega_star * t_shifted) - \
                                                                                B * self.alpha * np.sin(omega_star * t_shifted))
                    w_total[t_index] += w_mn[t_index, m - 1, n - 1] * self.phi_mn(m, n)
                    print("t_shifted: ", t_shifted, "|| curr t: ", current_time, "|| curr imp t: ", self.impulse_times[curr_impulse_index], "|| curr imp ind: ", curr_impulse_index, " || m: ", m, "|| n :", n, "|| A: ", A, "|| B: ", B, "|| w:", w_total[t_index, m - 1, n - 1])

        return t, w_total, w_mn, w_mn_dot_minus, w_mn_dot, self.x, self.y, self.X, self.Y, self.a, self.b, Q, omega_resonant