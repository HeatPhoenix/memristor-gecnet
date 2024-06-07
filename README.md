# Backward Generation Optimal Samples

This repository contains the code used to write the paper: "[Neural representation of a time optimal, constant acceleration rendezvous](https://arxiv.org/abs/2203.15490)"

A video is also provided explaining the Guidance & Control Networks (G&CNETs) and the Backward propagation of optimal samples: "[Video: Neural representation of a time optimal, constant acceleration rendezvous](https://youtu.be/XdpqDP_hY4k)"  

## Description

We train two neural models using the data augmentation technique called backward generation of optimal examples. The models learn to represent the optimal control policy (i.e. thrust direction) and the value function (i.e. time of flight) for a time-optimal, constant acceleration rendezvous. The optimal control problem is inspired by the 11th Edition of the Global Trajectory Optimisation Competition (GTOC 11) and constitutes transfers starting from the asteroid belt and ending at a specific phase of a circular orbit.

## Repository structure

For each of the two models, it is recommended to run the notebooks in the order in which they are listed below.

### Optimal control policy notebooks:
#### 1 - The nominal trajectory
Creates one nominal trajectory using a single shooting method. Saves the result in a pickled file.
#### 2 - Backward Generation of Optimal Samples
Loads a nominal trajectory (from the pickled file) and creates a dataset containing a bundle of trajectories around the nominal (backward generation of optimal examples method). Saves the results in a pickled file.
#### 3 - Train a G&CNET
Trains a feedforward neural network to represent the optimal control policy using the pickled dataset generated in the previous notebook. Saves the loss plot (as figure and pickle) and the model weights.
#### 4 - Validate a G&CNET
Creates error plots for thrust direction using a test set (which is generated using 2 - Backward Generation of Optimal Samples) and saves the figures.
#### 5 - Evaluate a G&CNET transfer
Evaluates the model by simulating the system dynamics using the control input provided by the trained model. Hence one can assess whether the learned control accumulates errors over an entire transfer or not. The dynamics are integrated until the nominal transfer time and until the semi-major axis is reached. The mean errors in final position, velocity, equinoctial parameters and time are computed (using a test set) as well as average prediction errors.
### Value function notebooks:
#### 6_1 - Value function learning - Nominal trajectories
Generates a pickled dataset which contain initial conditions starting from the asteroid belt and the corresponding solutions found by the single shooting method. The notebook is written such as to reduce the amount of local minima injected into the training dataset by solving each trajectory numerous times (starting from different initial guesses).
#### 6_2 - Learn from nominal database
Loads the pickled nominal trajectories and trains a simple feedforward neural network to represent the time of flight given the system's initial conditions in the asteroid belt. Creates a pickled test file.
#### 6_3 - Learn from the augmented database
Augments the pickled nominal trajectories using the backward generation of optimal examples and trains a feedforward neural network to represent the time of flight. Creates pickled test/training files and a figure comparing the predictions made by a network trained with and without augmentation.

### Folders:

#### data 
Contains the pickled files produced by various notebooks (Note that the actual training dataset is not committed and must be regenerated running the corresponding notebooks)
#### figures
Contains the figures produced for the paper
#### mycore
A python module used for the main common classes
#### scripts
Contains python file used to generate the training set for the value function learning on the supercomputer (Note that the training dataset can also be generated with 6_1 - Value function learning - Nominal trajectories).
#### seb
Legacy code

## Dependencies

The following libraries are needed to run the notebooks (we recommend to use conda to avoid clashes during installation). The versions listed below are the ones that were used to generate the results, but other versions should also work.
```
numpy == 1.21.3
heyoka.py == 0.17.0
pykep == 2.6
matplotlib == 3.4.3
pygmo == 2.11.3
pygmo_plugins_nonfree == 0.10
pickle == 0.0.12
pandas == 1.3.4
time == 1.8
tqdm == 4.62.3
pytorch == 1.11.0
jupyter == 1.0.0
ipython == 7.32.0
scikit-learn == 1.0.2
scipy == 1.7.1
```

**Important note for notebooks:**
- 1 - The nominal trajectory
- 6_1 - Value function learning - Nominal trajectories

To run the single shooting method in these notebooks, we used the sequential quadratic programming (SQP) solver SNOPT (the corresponding library files are needed). Other SQP solvers may also work.
