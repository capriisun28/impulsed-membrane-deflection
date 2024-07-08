# Impulsed Membrane Response
*A Purdue SURF 2024 project*

README added by Adam; probably good to have this here. 

## Notes from Carolyn:

### Motivation:

This project builds upon the work of previous undergraduate students with the Webb group (Scott Kenning and Will Pavlick). Previous implementations employed finite element and exact analytical methods to solve the partial differential equation describing membrane deflection in response to a source pressure, implemented simulations of membrane deflection in response to a constant and single frequency pressure function. 

With this model, I simulate membrane responses to an arbitrary pressure function with specified spatial and temporal profiles. This is done via considering membrane vibrations through time, apropos of incident pulses that are effectively instantanous (modelled as dirac delta functions.) Essentially, an instantanous incident pulse (mimcking pulses from a laser beam) serves to impart a momentum to the membrane that drives the membrane to vibrate; in between pulses the membrane freely oscillates. 

 This will be useful for the Webb group–having the ability to simulate the response of a membrane to any arbitrary pressure function will allow for more streamlined setup and design for future experiments. Avenues that might produce interesting results can be much more readily and cost-effectively explored first through simulation, before deploying an experiment. In general, it is useful to have a robust understanding of the physics underlying optomechanical membrane deformations, as there are many useful applications of optomechanical systems, from implementing effective switches in all-optical communications systems, to optical tweezers that trap atoms and nanoparticles and have applications in fields ranging from biology to microfluidics to quantum computing.

### How to run the sims:
The main script in question is model.py (this may change as I continue refactoring code). To run this script, the repo can be cloned locally, or one could download model.py and in the correct directory in the terminal's command line, type "python3 model.py". If this doesn't work, try "python model.py".
If numpy/matplotlib are not yet installed such that running the above command results in an error that looks like: "Import error: No module named numpy" or "Import error: No module named matplotlib.pyplot", the respective libraries can be installed by running "pip3 install numpy" or "pip3 install matplotlib" (if this throws a "pip3 command not found" error, try just "pip" instead of "pip3").

### What the sims plot:
(If I can get things to work again. AGHHHHHHHHHHHHHH ........ )

plot_displacement: Shows a 2d overview of the membrane and its displacement response at each time step.

plot_displacement_vs_time & plot_avg_displacement_vs_time: Plotting just the magnitude of the displacement and the average magnitude of the displacement as a function of time. (not sure if only one is needed, I have both for now)

plot_cutout_along_plane: Picking a cutout line that runs across the membrane, and plotting the displacement along that cutout

plot_individual_modes: Plotting the individual modes of the displacement response. In this case, only w_1_1 is being plotted as a sanity check.

### What I'd still like to implement:

- Being able to define multiple source pressures, and superimposing the force they impart to the membrane/the responses of the membrane wrt each individual force
- Being able to define source pressures with spatial profiles that are not only rectangles (perhaps radial)
- To test: different known membrane responses by plugging in proper parameters to define the membrane

---

## Adam's Note on Impulse "Handling"

See the file `adam-note_impulses_2024-06-28.txt` for some thoughts on
how impulses are handled.

I'd suggest focusing on implementing the impulse handling for a
single spatial mode coefficient (e.g., w_11) and plotting its value
over time, to make sure that's being handled properly. Then once
that's working, move back to plotting the whole membrane response.

