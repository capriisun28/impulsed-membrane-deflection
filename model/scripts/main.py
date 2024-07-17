from membrane_response import membrane_response
from plots import plotting

print("hi!") #easy flag to scroll to the start

# Note: at least 1 impulse_times array must be uncommented
#       and one initialization of a "deflection" instance uncommented

# test case 1 (ideal parameters, velocity vs time graph looks weird, continuity on displacement graph also looks odd)

#impulse_times = [0]
impulse_times = [0, 1e-6, 2e-6, 3e-6]
#impulse_times = [0, 9e-6, 2e-5, 3.8e-5]
deflection = membrane_response(impulse_times, dt=1e-7, t_max=5e-5, eta=10, modes=100)

# test case 2 (used for easier debugging purposes (less terminal output))
#impulse_times = [0.6, 1, 2]
#deflection = membrane_response(impulse_times, dt=0.5, t_max=2.51, h=5e-4, eta=2)
t, w_total, w_mn, w_mn_dot_minus, w_mn_dot, x, y, X, Y, a, b, Q, resonant_freq = deflection.calculate_response()


# plotting the results  #the plots take a bit to run
# this plot should be the main membrane response
plotting.plot_displacement(w_total, t, a, b)

# calls the plotting fn for displacement v time
# commented out for now as this runs really slowly
#plotting.plot_displacement_vs_time(w_total, t, x, y)

# calls the plotting fn for avg displacement v time
#plotting.plot_avg_displacement_vs_time(w_total, t)

# calls the plotting fn for cutout of the response
cutout_line = (deflection.b)/2
#plotting.plot_cutout_along_plane(t, x, y, w_total, plane='y', value=cutout_line)
plotting.plot_cutout_along_plane_at_timesteps(t, x, y, w_total, plane = 'y', value=cutout_line)

# uncomment to check individual modes of the displacement response
#plotting.plot_individual_modes(w_mn, t, 0, 0)
#plotting.plot_velocity_imparted_over_time(w_mn_dot_minus, t, 0, 0)
#plotting.plot_velocity_of_mode_over_time(w_mn_dot, t, 0, 0)

#plotting.plot_Q(Q)
#plotting.plot_omega_res(resonant_freq)
