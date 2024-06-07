#!/usr/bin/env python
# coding: utf-8


#get_ipython().system('export PYTHONPATH=$PYTHONPATH:./aihwkit/src/')

# Core imports

from math import e
import sys
sys.path.insert(0,'/home/heatphoenix/aihwkit_pr/aihwkit/src')
#from aihwkit_pr.aihwkit.src.aihwkit.nn.modules.linear_sliced import AnalogLinearBitSlicing
from aihwkit.nn import AnalogLinearBitSlicing
from mycore import ast2station_rotating, ast2station_rotating2


# PyTorch and ML
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

# Main imports
import heyoka as hy
import pykep as pk
import pandas as pd

# Usual imports
import time
import numpy as np
import pickle as pkl
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import copy

# Args import
import argparse

# # Setting up IBM AI HW Kit
# Imports from PyTorch.
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
import torch.nn.functional as F

# Imports from aihwkit.
from aihwkit.optim.analog_optimizer import AnalogAdam
from aihwkit.nn import AnalogConv1d, AnalogConv2d, AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import SingleRPUConfig, ConstantStepDevice
from aihwkit.simulator.rpu_base import cuda
from aihwkit.simulator.parameters import WeightNoiseType
from aihwkit.simulator.configs import InferenceRPUConfig, WeightModifierType
from aihwkit.inference import ReRamWan2022NoiseModel, PCMLikeNoiseModel
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator.tiles.base import AnalogTileStateNames, BaseTile, TileModuleBase
from aihwkit.exceptions import TileModuleError

import datetime
import csv   

# %%
# Core imports
from mycore import ast2station_rotating, ast2station_rotating2, heyoka_ffnn, build_taylor_ffnn

# PyTorch and ML
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Main imports
import heyoka as hy
import pykep as pk
import pandas as pd

# Usual imports
import time
import numpy as np
from scipy.integrate import odeint
import pickle as pkl
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D

# Get the current date and time
current_time = datetime.datetime.now()

# Convert the date and time to a string
timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")

# Print the timestamp
print("Run Date Timestamp:", timestamp)

# Store the timestamp in a variable
timestamp_str = str(timestamp)

# Create the parser
parser = argparse.ArgumentParser(description='Run a HW G&CNET and examine drift')

# Add an argument
parser.add_argument('--neurons', type=int, help='Number of neurons (default=128)')
parser.add_argument('--pcm', action='store_true', help='Use PCM (default=False)')
parser.add_argument('--rram', action='store_true', help='Use RRAM (default=False)')
parser.add_argument('--fault_ratio', type=float, help='Fault ratio (default=0)')
parser.add_argument('--slices', type=int, help='Slices per weight (default=1)')
parser.add_argument('--epochs', type=int, help='Number of epochs to train (default=300)')
parser.add_argument('--drift', type=float, help='Inference time (for drift simulation)')
parser.add_argument('--adc_res', type=int, help='ADC resolution (default=inf)')
parser.add_argument('--batch_size', type=int, help='Batch size (default=4096)')
parser.add_argument('--digital', action='store_true', help='Use digital network (default=False)')
parser.add_argument('--learning_rate', type=float, help='Learning rate (default=0.5e-4)')
parser.add_argument('--gpu', type=int, help='GPU to use (default=7)')


# Parse the arguments
args = parser.parse_args()

#Parameters from command line args
pcm = args.pcm 
rram = args.rram
slices = args.slices if args.slices else 1
fault_ratio = args.fault_ratio if args.fault_ratio else 0.0
drift = args.drift if args.drift else 0.0
adc_res = args.adc_res if args.adc_res else float('inf')
neurons = args.neurons if args.neurons else 128  # Default to 128 if no argument is provided
n_epochs = args.epochs if args.epochs else 300
batch_size_train = args.batch_size if args.batch_size else 4096
learning_rate = args.learning_rate if args.learning_rate else 0.5e-4
chosen_gpu = args.gpu if args.gpu else 0
digital = args.digital

if drift != 0.0 and drift != 1.0 and drift != 3600*24.0 and drift != 3600*24.0*2:
    raise ValueError("Drift time must be 0.0, 1.0, 3600*24.0 (1 day) or 3600*24.0*2 (2 days)")

torch.backends.cudnn.benchmark = True

#Hardware configuration
rpu_config = InferenceRPUConfig()
#important for HW-aware training
rpu_config.backward.is_perfect = True

#RRAM configuration
rram_inference_config = InferenceRPUConfig()
rram_inference_config.modifier.pcm_prob_at_gmax = fault_ratio/2 #0.05 
rram_inference_config.modifier.pcm_prob_at_reset = fault_ratio/2
rram_inference_config.forward.w_noise_type = WeightNoiseType.ADDITIVE_CONSTANT
rram_inference_config.forward.w_noise = 0.010 #0.01

#PCM configuration
pcm_config = InferenceRPUConfig()
pcm_config.noise_model = PCMLikeNoiseModel(g_max=25.0)
pcm_config.forward.w_noise_type = WeightNoiseType.ADDITIVE_CONSTANT
pcm_config.forward.w_noise = 0.02
pcm_config.modifier.pcm_prob_at_gmax = fault_ratio/2 #0.05 
pcm_config.modifier.pcm_prob_at_reset = fault_ratio/2

if pcm:
    rpu_config = pcm_config
    print("Using PCM")
elif rram:
    rpu_config = rram_inference_config
    print("Using RRAM")
elif digital:
    print("Using digital")
else:
    raise ValueError("Please choose a config, --pcm or --rram")


if not digital: 
    print(f"Using {neurons} neurons, PCM: {pcm}, RRAM: {rram}, slices: {slices}, fault_ratio: {fault_ratio}, drift: {drift}, adc_res: {adc_res}")
else:
    print(f"Using {neurons} neurons, {n_epochs} epochs, batch size: {batch_size_train}, learning rate: {learning_rate}, digital network")

if adc_res != float('inf'):
    rpu_config.forward.inp_res = adc_res**2
    rpu_config.forward.out_res = adc_res**2

type_str = 'pcm' if pcm else 'rram' if rram else 'digital'
constr_filename =  f"{n_epochs}_epochs_4_layers_{neurons}_neurons_{fault_ratio}_fault_{slices}_slices_{batch_size_train}_batch_size_{type_str}_loss2.mdl" #_{timestamp_str}
print(f"Contructed filename: {constr_filename}")


if torch.cuda.is_available():
    print("PyTorch detected CUDA")
    device = torch.device("cuda:" + str(chosen_gpu))
else: 
    print("PyTorch DID NOT detect CUDA")
    device = torch.device("cpu")
print("Training on :", device)


# Kaiming_normal initialization method
def initial_weights(m):
    """
    - Initializes weights in each layer according to 'He initialization'.
    Args:
        m   (torch.nn.Linear):         - Neural network layers
    Returns:
        None
    """
    if type(m) == torch.nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight)

# # Optimizer and Loss
# The loss is the cosine of the angle between the predicted and the actual thrust vector
cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-8)
# This first loss is directly in degrees
def loss1(ground_truth, prediction):    
    return torch.mean(torch.arccos(cosine_similarity(ground_truth, prediction))) / np.pi * 180
# This is the loss used in the paper with Ekin
def loss2(ground_truth, prediction):    
    return 1 - torch.mean(cosine_similarity(ground_truth, prediction))
loss = loss2

# Lists used to plot loss over epochs
loss_list = []
loss_val_list = []


# # Training loop
batch_size_val = 4096

model = AnalogSequential(
        AnalogLinearBitSlicing(6, neurons, slices, True, rpu_config=rpu_config),
        nn.Softplus(),
        AnalogLinearBitSlicing(neurons, neurons, slices, True, rpu_config=rpu_config),
        nn.Softplus(),
        AnalogLinearBitSlicing(neurons, neurons, slices, True, rpu_config=rpu_config),
        nn.Softplus(),
        AnalogLinearBitSlicing(neurons, 3, slices, True, rpu_config=rpu_config),
        )
model.load_state_dict(torch.load(f"models/{constr_filename}"))
state_dict = torch.load(f"models/{constr_filename}")


class MaskLayer(nn.Module):
    def __init__(self, output_shape, probability, device):
        super(MaskLayer, self).__init__()
        self.mask = torch.Tensor(output_shape).bernoulli_(1-probability).to(device)
        self.output_shape = output_shape

    def forward(self, x):
        return x * self.mask
    
    def update_mask(self, new_probability):
        self.probability = new_probability
        self.mask = torch.Tensor(self.output_shape).bernoulli_(1-self.probability).to(device)

    def __str__(self):
        return f"MaskLayer(output_shape={self.output_shape}, probability={self.probability})"

# Find indices of Softplus layers
softplus_indices = [i for i, layer in enumerate(model.children()) if isinstance(layer, nn.Softplus)]

# Insert MaskLayer after each Softplus layer
for i in reversed(softplus_indices):
    mask_layer = MaskLayer(model[i-1].out_features, fault_ratio, device)
    model.insert(i + 1, mask_layer)

if fault_ratio > 0:
    print(f"Masked with {fault_ratio}")

model = model.cuda(device)
model.eval()

with open("data/test_4_189_250_100_[0.001, 0.001, 0.001, 0.001]_[1.0, 1.0, 1.0, 1.0].pk", "rb") as file:
    data = pkl.load(file)
# %%
# This was defined early on in other notebooks and here hardcoded (careful)
r_target = 1.3 * pk.AU
# Problem data
GAMMA = 1e-4
OMEGA = np.sqrt(pk.MU_SUN/r_target**3) 

# %%
traj_id = 34
# We load the data for one particular trajectory
state = data[traj_id][:,:6]
thrust = data[traj_id][:,6:9]
ic = state[-1]
print(ic)
tof  = data[traj_id][0,-1]

# %% [markdown]
# Using scipy integrator (good enough for this as precision required is low)

# %%
log_ix = []
log_iy = []
log_iz = []
def dxdt(x, t, model):
    L = pk.AU
    TIME = np.sqrt(L**3/pk.MU_SUN)   # Unit for time (period)
    ACC = L/TIME**2                  # Unit for accelerations
    # Non-dimensionalize:
    mu = 1.
    ndGAMMA = GAMMA / ACC
    ndOMEGA = OMEGA * TIME
    #print(type(x))
    x_tensor = torch.tensor(x).to(device)
    inp = x_tensor.float()
    #print(inp)
    ix, iy, iz = model(inp).detach()
    ix = float(ix.cpu().numpy())
    iy = float(iy.cpu().numpy())
    iz = float(iz.cpu().numpy())
    norm = np.sqrt(ix**2 + iy**2 + iz**2)
    ix = ix / norm
    iy = iy / norm
    iz = iz / norm
    
    log_ix.append(ix)
    log_iy.append(iy)
    log_iz.append(iz)

    dxdt = [0]*6
    dxdt[0] = x[3]
    dxdt[1] = x[4]
    dxdt[2] = x[5]
    dxdt[3] = -mu*x[0]/((x[0]**2+x[1]**2+x[2]**2)**(3/2)) + 2*ndOMEGA*x[4] + ndOMEGA**2*x[0] + ndGAMMA*ix
    dxdt[4] = -mu*x[1]/((x[0]**2+x[1]**2+x[2]**2)**(3/2)) - 2*ndOMEGA*x[3] + ndOMEGA**2*x[1] + ndGAMMA*iy
    dxdt[5] = -mu*x[2]/((x[0]**2+x[1]**2+x[2]**2)**(3/2)) + ndGAMMA*iz
    return dxdt


# %%
tgrid = np.linspace(0, tof, 100)
scipy_sol = odeint(dxdt, ic, tgrid, args=(model,), rtol=1e-9,atol=1e-9)
err_r = np.linalg.norm(scipy_sol[-1,:3] - state[0,:3])
err_v = np.linalg.norm(scipy_sol[-1,3:6] - state[0,3:6])
print(f"r: {err_r}, v: {err_v}")

# Save the loss trend

with open("log_i.pk","wb") as file:
    pkl.dump([log_ix, log_iy, log_iz], file)



# %%
err_r = np.linalg.norm(scipy_sol[-1,:3] - state[0,:3])
err_v = (np.linalg.norm(scipy_sol[-1,3:6] - state[0,3:6]))
r = scipy_sol[-1,:3] * pk.AU
v = scipy_sol[-1,3:6] * np.sqrt(pk.MU_SUN/pk.AU)
# We move the velocity to the inertial frame
v[0] = v[0] - OMEGA * r[1]
v[1] = v[1] + OMEGA * r[0]
# Convert to equinoctial
E = pk.ic2eq(r,v, pk.MU_SUN)
ecc_final = np.sqrt(E[1]**2+E[2]**2)
err_a = (E[0]/(1-ecc_final**2) - r_target)
err_e = (np.sqrt(E[1]**2+E[2]**2))
err_i = (2*np.arctan(np.sqrt(E[3]**2+E[4]**2)))

print(f"Error on final position  (AU): {np.mean(err_r)*pk.AU/pk.AU}")
print(f"Error on final velocity  (km/s): {np.mean(err_v)*np.sqrt(pk.MU_SUN/pk.AU)/1000.}")
print(f"Error on final sma  (AU): {np.mean(err_a)/pk.AU}")
print(f"Error on final ecc: {np.mean(err_e)}")
print(f"Error on final incl  (degrees): {np.mean(err_i)*pk.RAD2DEG}")

# def build_taylor_ffnn(L, MU, GAMMA, OMEGA, tol=1e-16):
#     """Builds an integrator for the state equation of the optimal transfer with constant acceleration
#       in a rotating frame (axis z). The Thrust direction is given by a ffnn
#     Args:
#         L (float): units for length (in m)
#         MU (float): units for the gravitational parametrs (in kg m^2/s^3)
#         GAMMA (float): constant acceleration (in N)
#         OMEGA (float): angular velocity (in rad/sec)
#     Returns:
#         [hy.taylor_adaptive]: the adaptive integartor with state
#     """
#        # Unit definitions:
#     TIME   = np.sqrt(L**3/MU)                           # Unit for time (period)
#     ACC    = L/TIME**2                                  # Unit for accelerations
    
#     # Non-dimensionalize:
#     mu = 1.
#     GAMMA = GAMMA / ACC
#     OMEGA = OMEGA * TIME
    
#     # Create symbolic variables
#     x, y, z, vx, vy, vz = hy.make_vars("x", "y", "z", "vx", "vy", "vz")

#     # Optimal thrust angle theta and phi
#     inputs = [ x, y, z, vx, vy, vz]

#     # GCNET
#     linear = lambda inp: inp
#     softplus = lambda inp2: hy.log(1+hy.exp(inp2))
#     ffnn = hy.model.ffnn(inputs = inputs, nn_hidden = [128*16, 128*16, 128*16], n_out = 3, activations = [softplus, softplus, softplus, linear])#, nn_wb = ffnn_nn_wb) # OG


#     ix, iy, iz = ffnn

#     norm_outputs = hy.sqrt(ix**2+iy**2+iz**2)
#     ix = ix / norm_outputs
#     iy = iy / norm_outputs
#     iz = iz / norm_outputs
#     # Create Taylor integrator
#     ta = hy.taylor_adaptive(sys = [
#         (x, vx),
#         (y, vy),
#         (z, vz),
#         (vx, -mu*x/((x**2+y**2+z**2)**(3/2)) + 2*OMEGA*vy + OMEGA**2*x + GAMMA*ix),
#         (vy, -mu*y/((x**2+y**2+z**2)**(3/2)) - 2*OMEGA*vx + OMEGA**2*y + GAMMA*iy),
#         (vz, -mu*z/((x**2+y**2+z**2)**(3/2)) + GAMMA*iz),
#         ],
#         # Initial conditions:
#         state = [1., 1., 1., 1., 1., 1.],
#         # Initial value of time variable:
#         time = 0.,
#         compact_mode=True,
#         tol=tol)
#     return ta


# with open("data/nominal_E_4_189.pk", "rb") as file:
#     x0, r_target, champion_x = pkl.load(file)

# x0 = np.array(x0)
# tf_opt = champion_x[-1]

# # This was defined early on in other notebooks and here hardcoded (careful)
# r_target = 1.3 * pk.AU
# # Problem data
# GAMMA = 1e-4
# OMEGA = np.sqrt(pk.MU_SUN/r_target**3) 
# MU = pk.MU_SUN
# L = pk.AU

# # Heyoka
# ta_eom = build_taylor_ffnn_siren(L, MU, GAMMA, OMEGA, tol=1e-16)
# weights_ls = [state_dict[key].cpu() for key in state_dict.keys() if 'weight' in key]
# bias_ls = [state_dict[key].cpu() for key in state_dict.keys() if 'bias' in key]
# weights_biases = weights_ls + bias_ls
# flattened_nw = np.concatenate(tuple(wb.flatten() for wb in weights_biases))
# ta_eom.pars[:] = flattened_nw


# ta_eom.state[:] = ic
# ta_eom.time = 0.
# tgrid = np.linspace(0, tf_opt, 100)
# sol = ta_eom.propagate_grid(tgrid)[5]