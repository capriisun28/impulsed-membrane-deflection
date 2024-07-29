import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.colors as mcolors

def checkpath(output_path = 'output_plots/', displacement=False, cutout=False):
        script_dir = os.path.dirname(__file__)
        results_dir = os.path.join(script_dir, output_path)
        #displacement_plots = os.path.join(results_dir, output_path)
        #displacement_cutout_plots = os.path.join(results_dir, output_path)

        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
            #os.makedirs(displacement_plots)
            #os.makedirs(displacement_cutout_plots)

        if displacement:
            return checkpath(output_path=results_dir + "displacement_plots/")
        elif cutout:
            return checkpath(output_path=results_dir + "displacement_cutout_plots/")
        else:
            return results_dir

class plotting:

    @staticmethod
    def plot_displacement(w_total, t, a, b):
        #######################################
        # ADDED BY ADAM: Get max value for limits
        cbar_max_val = w_total.max()
        cbar_min_val = w_total.min()
        if (abs(cbar_min_val) > cbar_max_val):
            cbar_max_val = abs(cbar_min_val)
        #######################################
        for ti in range(0, len(t), int(len(t)/100)): # adjust later, have the factor that len(t) is divided by depend on time steps
            fig = plt.figure(figsize=(10, 6)) # Make a new figure with specified size for every plot (will help make saved figure PNGs consistent)
            #levels = 20
            #plt.contourf(self.x, self.y, w_total[ti], cmap='magma', vmin=-cbar_max_val, vmax=cbar_max_val)
            # It appears that imshow might be a better fit for this. The "vmin" and "vmax" kwargs let you set colorbar limits
            # The "extent" specifies the axis tick label ranges
            plt.imshow(w_total[ti], cmap='magma', interpolation='nearest', extent=[0,a,0,b], vmin=-cbar_max_val, vmax=cbar_max_val)
            plt.colorbar(label = 'Displacement (m)')
            plt.title(f'Membrane Displacement Response at Time = {t[ti]:.2e} s') # Changed format specification to scientific notation
            plt.xlabel('x (m)')
            plt.ylabel('y (m)')
            plt.savefig(checkpath(displacement=True) + f'displacement_{ti:03d}.png', format="png") # Changed format specification to have leading zeroes so ImageMagick plays nice with it
            #plt.show()
            plt.close(fig)

    def plot_displacement_for_poster(w_total, t, a, b):
        #######################################
        # ADDED BY ADAM: Get max value for limits
        cbar_max_val = w_total.max()
        cbar_min_val = w_total.min()
        if (abs(cbar_min_val) > cbar_max_val):
            cbar_max_val = abs(cbar_min_val)
        #######################################
        for ti in range(0, len(t), int(len(t)/100)): # adjust later, have the factor that len(t) is divided by depend on time steps
            fig = plt.figure(figsize=(10, 6)) # Make a new figure with specified size for every plot (will help make saved figure PNGs consistent)
            #levels = 20
            #plt.contourf(self.x, self.y, w_total[ti], cmap='magma', vmin=-cbar_max_val, vmax=cbar_max_val)
            # It appears that imshow might be a better fit for this. The "vmin" and "vmax" kwargs let you set colorbar limits
            # The "extent" specifies the axis tick label ranges
            plt.imshow(w_total[ti], cmap='magma', interpolation='nearest', extent=[0,a,0,b], vmin=-cbar_max_val, vmax=cbar_max_val)
            plt.colorbar(label = 'Displacement (m)')
            plt.title(f'Time = {t[ti]:.2e} s') # Changed format specification to scientific notation
            plt.savefig(checkpath(displacement=True) + f'displacement_{ti:03d}.png', format="png") # Changed format specification to have leading zeroes so ImageMagick plays nice with it
            #plt.show()
            plt.close(fig)

    def plot_point_deflection_vs_time(w_total, t, x, y, a, b):
        #convert (x, y) coordinates to indices
        x_idx = int((x / a) * (w_total.shape[1] - 1))
        y_idx = int((y / b) * (w_total.shape[2] - 1))
        deflection_at_point = w_total[:, x_idx, y_idx]
        plt.figure()
        plt.plot(t, deflection_at_point, label=f'Point ({x:.3e}, {y:.3e})', color=mcolors.CSS4_COLORS.get("darkslateblue"))
        plt.xlabel('Time (s)')
        plt.ylabel('Deflection (m)')
        plt.title(f'Deflection vs Time at Point ({x}, {y})')
        plt.legend()
        plt.savefig(checkpath() + f'deflection_point_{x}_{y}_vs_time.png', format="png")
        #plt.show()
        plt.close()
    
    def plot_displacement_vs_time(w_total, t, x, y):
        plt.figure()
        for i in range(0, len(x)):
            for j in range(0, len(y)):
                plt.plot(t, w_total[:, i, j], label=f'Point ({x[i]:.3e}, {y[j]:.3e})')
        plt.xlabel('Time (s)')
        plt.ylabel('Displacement at Center (m)')
        plt.title('Displacement vs Time at Center of Membrane')
        #plt.legend()
        plt.savefig(checkpath() + 'displacement_vs_time.png', format="png")
        #plt.show()
        plt.close()

    
    def plot_avg_displacement_vs_time(w_total, t):
        avg_displacement = np.mean(w_total, axis=(1, 2))
        plt.figure()
        plt.plot(t, avg_displacement)
        plt.xlabel('Time (s)')
        plt.ylabel('Average Displacement (m)')
        plt.title('Average Displacement vs Time')
        plt.savefig(checkpath() + f'avg_displacement_v_time.png', format="png")
        #plt.show()
        plt.close()
    
    def plot_cutout_along_plane(t, x, y, w_total, plane='y', value=0.5):
        plt.figure()
        if plane == 'y':
            idx = (np.abs(y - value)).argmin()
            for ti in range(len(t)):
                plt.plot(x, w_total[ti, idx, :], label=f'Time = {t[ti]:.2f}s') # fix label
            plt.xlabel('Distance along membrane (x) (m)')
        elif plane == 'x':
            idx = (np.abs(x - value)).argmin()
            for ti in range(len(t)):
                plt.plot(y, w_total[ti, :, idx], label=f'Time = {t[ti]:.2f}s')
            plt.xlabel('Distance along membrane (y) (m)')
        plt.ylabel('Displacement (w) (m)')
        plt.title(f'Displacement along {plane}={value}')
        #plt.legend()
        plt.savefig(checkpath() + f'displacement_{plane}_{value}_cutout_{ti}.png', format="png")
        #plt.show()
        plt.close()
        """    
        else:
            raise ValueError("Only 'y' plane cutout is implemented.")
        """

    def plot_cutout_along_plane_at_timesteps(t, x, y, w_total, plane='y', value=0.5):
        if plane == 'y':
            idx = (np.abs(y - value)).argmin()
        elif plane == 'x':
            idx = (np.abs(x - value)).argmin()
        else:
            raise ValueError("Only 'x' and 'y' planes are supported.")
        
        cbar_max_val = w_total.max()
        cbar_min_val = w_total.min()
        if abs(cbar_min_val) > cbar_max_val:
            cbar_max_val = abs(cbar_min_val)

        global_min = np.inf
        global_max = -np.inf
        for ti in range(len(t)):
            if plane == 'y':
                data = w_total[ti, idx, :]
            elif plane == 'x':
                data = w_total[ti, :, idx]
            global_min = min(global_min, data.min())
            global_max = max(global_max, data.max())
        
        for ti in range(0, len(t), max(1, int(len(t) / 100))):  # adjust factor depending on the number of time steps
            fig = plt.figure(figsize=(10, 6))
            
            if plane == 'y':
                plt.plot(x, w_total[ti, idx, :], label=f'Time = {t[ti]:.3e}s')
                plt.xlabel('Distance along membrane (x) (m)')
            elif plane == 'x':
                plt.plot(y, w_total[ti, :, idx], label=f'Time = {t[ti]:.3e}s')
                plt.xlabel('Distance along membrane (y) (m)')
            
            plt.ylabel('Displacement (w) (m)')
            plt.ylim(global_min, global_max)
            plt.title(f'Displacement along {plane}={value} at Time = {t[ti]:.2e}s')
            #plt.legend()
            plt.savefig(checkpath(cutout=True) + f'displacement_{plane}_{value}_timestep_{ti:03d}.png', format="png")
            #plt.show()
            plt.close(fig)

    def plot_individual_modes(w_mn, t, m, n):
        plt.figure()
        plt.plot(t, w_mn[:, m - 1, n - 1])
        plt.xlabel('Time (s)')
        plt.ylabel(f'Mode {m},{n} Displacement')
        plt.title(f'Displacement of Mode ({m},{n}) vs Time')
        plt.legend()
        plt.savefig(checkpath() + f'individual_mode_{m}_{n}_over_time.png', format="png")
        #plt.show()
        plt.close()
    
    def plot_individual_modes_together(w_mn, t, m1, n1, m2, n2, m3, n3):
        plt.figure()
        plt.plot(t, w_mn[:, m1 - 1, n1 - 1], label=f'Mode ({m1},{n1})', color=mcolors.CSS4_COLORS.get("orchid"))
        plt.plot(t, w_mn[:, m2 - 1, n2 - 1], label=f'Mode ({m2},{n2})', color=mcolors.CSS4_COLORS.get("lightseagreen"))
        plt.plot(t, w_mn[:, m3 - 1, n3 - 1], label=f'Mode ({m3},{n3})', color=mcolors.CSS4_COLORS.get("mediumslatepurple"))
        plt.ylabel('Displacement')
        plt.title('Displacement of Modes vs Time')
        plt.legend()
        plt.savefig(checkpath() + 'combined_modes_over_time.png', format="png")
        #plt.show()
        plt.close()
    
    def plot_velocity_imparted_over_time(w_mn_dot_minus, t, m, n):
        plt.figure()
        plt.plot(t, w_mn_dot_minus[:, m - 1, n - 1])
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity Imparted (m/s)')
        plt.title(f'Velocity Imparted Over Time for Mode ({m}, {n})')
        plt.legend()
        plt.savefig(checkpath() + f'velocity_mode_{m}_{n}_imparted_over_time.png', format="png")
        #plt.show()
        plt.close()

    
    def plot_velocity_of_mode_over_time(w_mn_dot,t, m, n):
        plt.figure()
        #plt.plot(t, w_mn_dot[:, 1, 1])
        plt.plot(t, w_mn_dot[:, m - 1, n - 1])
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (m/s)')
        plt.title(f'Velocity Over Time for Mode ({m}, {n})')
        plt.savefig(checkpath() + f'velocity_for_mode_{m}_{n}_over_time.png', format="png")
        #plt.show()
        plt.close()

    def plot_Q(Q):
        Q = Q[1:, 1:]
        plt.figure(figsize=(8, 6))
        X, Y = np.meshgrid(range(Q.shape[1]), range(Q.shape[0]))
        cp = plt.contourf(X, Y, Q, cmap='magma')
        #plt.imshow(Q[1:,1:], cmap='magma', origin='lower', aspect='auto')
        plt.colorbar(cp, label='Q Value')
        plt.title("Q")
        plt.xlabel("mode #")
        plt.ylabel("mode #")
        plt.savefig(checkpath() + "q_contour_map_plotting.png", format='png')
        #plt.show()
        plt.close()

    def plot_omega_res(omega):
        omega = omega[1:, 1:]
        plt.figure(figsize=(8, 6))
        X, Y = np.meshgrid(range(omega.shape[1]), range(omega.shape[0]))
        cp = plt.contourf(X, Y, omega, cmap='magma')
        ### uncommenting out the belowline plots a heatmap as opposed to a contour map
        #plt.imshow(Q[1:,1:], cmap='magma', origin='lower', aspect='auto')
        plt.colorbar(cp, label='Resonant Frequency Value')
        plt.title("Resonant Frequency")
        plt.xlabel("mode #")
        plt.ylabel("mode #")
        plt.savefig(checkpath() + "resonant_freq_contour_map_plotting.png", format='png')
        #plt.show()
        plt.close()
