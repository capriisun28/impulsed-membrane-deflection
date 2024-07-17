# Impulsed Membrane Response
*A Purdue SURF 2024 project*

README added by Adam; probably good to have this here. 

Carolyn: Thanks Adam!

## Notes from Carolyn:

### Motivation:

This project builds upon the work of previous undergraduate students with the Webb group (Scott Kenning and Will Pavlick). Previous implementations employed finite element and exact analytical methods to solve the partial differential equation describing membrane deflection in response to a source pressure, implemented simulations of membrane deflection in response to a constant and single frequency pressure function. 

With this model, I simulate membrane responses to an arbitrary pressure function with specified spatial and temporal profiles. This is done via considering membrane vibrations through time, apropos of incident pulses that are effectively instantanous (modelled as dirac delta functions.) Essentially, an instantanous incident pulse (mimcking pulses from a laser beam) serves to impart a momentum to the membrane that drives the membrane to vibrate; in between pulses the membrane freely oscillates. 

 This will be useful for the Webb group–having the ability to simulate the response of a membrane to any arbitrary pressure function will allow for more streamlined setup and design for future experiments. Avenues that might produce interesting results can be much more readily and cost-effectively explored first through simulation, before deploying an experiment. In general, it is useful to have a robust understanding of the physics underlying optomechanical membrane deformations, as there are many useful applications of optomechanical systems, from implementing effective switches in all-optical communications systems, to optical tweezers that trap atoms and nanoparticles and have applications in fields ranging from biology to microfluidics to quantum computing.

### How to run the sims:

The main scripts in question may be found in model/scripts/. There are currently three python scripts:

**main.py**:
The script to be run. Plotting functions can be called/commmented out as wished. Custom values can be set as functions are called by providing a valid key-value pair in the function call.

To run this script, the repo can be cloned locally, or one could download model.py and in the correct directory in the terminal's command line, type "python3 main.py". If this doesn't work, try "python main.py".

If numpy/matplotlib are not yet installed such that running the above command results in an error that looks like: "Import error: No module named numpy" or "Import error: No module named matplotlib.pyplot", the respective libraries can be installed by running "pip3 install numpy" or "pip3 install matplotlib" (if this throws a "pip3 command not found" error, try just "pip" instead of "pip3").


**plots.py**:
Where all the plotting functions are stored.


**membrane_response.py**:
Where the logic and the actual dynamic forward modelling is implemented, wrt. to Adam's 05/23/24 "Membrane Impulse Response: Time Domain" notes.


### What the sims plot:

All functions are found in model/scripts/plots.py.

Plots of interest can be found in: model/output_plots/*relevant impulse folder*

Gifs can be found in:
- model/output_plots/{relevant impulse folder}/2d-displacement-plots 
- model/output_plots/{relevant impulse folder}/displacement-cutout-plots

**plot_displacement**: 
Shows a 2d overview of the membrane and its displacement response at each time step. These time steps pieced together comprise the gif animations shown in the group meeting, and are found in gif folders for various p0 initial conditions.

**plot_displacement_vs_time**: 
Plotting just the magnitude of the displacement as a function of time. Currently takes an abysmal amount of time to run, and the legend is quite gross, but this shall be remedied.

**plot_avg_displacement_vs_time**:
Plotting the average magnitude of the displacement as a function of time.

**plot_cutout_along_plane**: 
Picking a cutout line that runs across the membrane, and plotting the displacement along that cutout. Current default value is a horizontal line cutting halfway through the membrane (for a membrane with length 0.005m), but can be specified. Legend is also disgusting... for now!

**plot_cutout_along_plane_at_timesteps**:
Same as above, but plotted at each time steps also comprises the gifs shown in the meeting.

**plot_individual_modes**: 
Plotting the individual modes of the displacement response. In this case, only w_1_1 is being plotted as a sanity check.

**plot_velocity_imparted_over_time**:
The velocity the membrane is "kicked" to everytime it encounters an impulse. Mainly this plot is used to make sure that the velocity is updating correctly. When plotted, should look somewhat like a floor function with jumps, because plotted velocity values only change when an impulse is incident.

**plot_velocity_of_mode_over_time**:
For any specified mode, its velocity response is plotted against time.

### What I'd still like to implement:

- Being able to define multiple source pressures, and superimposing the force they impart to the membrane/the responses of the membrane wrt each individual force
- To test: different known membrane responses by plugging in proper parameters to define the membrane
- Further verifying that the model produces the correct behvaior

---

## Adam's Note on Impulse "Handling"

See the file `adam-note_impulses_2024-06-28.txt` for some thoughts on
how impulses are handled.

I'd suggest focusing on implementing the impulse handling for a
single spatial mode coefficient (e.g., w_11) and plotting its value
over time, to make sure that's being handled properly. Then once
that's working, move back to plotting the whole membrane response.

