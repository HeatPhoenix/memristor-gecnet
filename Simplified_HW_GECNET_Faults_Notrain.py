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
chosen_gpu = args.gpu if args.gpu else 7
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

# # Load the dataset
# The file nominal_bundle....pk is created in the previous notebook
datset_name="data/training_4_189_50000_100_[0.001, 0.001, 0.08, 0.08]_[1.0, 1.0, 1.0, 1.0].pk"
with open(datset_name, "rb") as file:
    data = pkl.load(file)
np.random.shuffle(data)


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

## Dataset split and preparation
# We split the whole dataset into training and testing
X_TRAIN, X_VALIDATION, Y_TRAIN, Y_VALIDATION = train_test_split(data.reshape(-1,10)[:,:6], data.reshape(-1,10)[:,6:9], train_size=0.8)
X_TRAIN = torch.tensor(X_TRAIN, device = device)
X_VALIDATION = torch.tensor(X_VALIDATION, device = device)
Y_TRAIN = torch.tensor(Y_TRAIN, device = device)
Y_VALIDATION = torch.tensor(Y_VALIDATION, device = device)


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

# Load random validation batch
X_VALIDATION, Y_VALIDATION = shuffle(X_TRAIN, Y_TRAIN)
states_val = X_VALIDATION[:batch_size_val, :]
labels_val = Y_VALIDATION[:batch_size_val, :]

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
model.to(device)
model.eval()

#drift_times = [0.0, 1.0, 1.0*3600*24, 1.0*3600*48]
fault_ratios = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 0.80]
for _ in range(12):
    for fault_ratio in fault_ratios:
        # Set the drift time
        
        for layer in model.children():
            if isinstance(layer, MaskLayer):
                layer.update_mask(fault_ratio)
                
        print(model)
        # Perform forward pass with the model
        predictions = model(states_val.float())
        
        # Compute the loss
        loss_val = loss(labels_val, predictions)
        
        # Append the loss to the list
        loss_list.append(fault_ratio)
        loss_val_list.append(loss_val.item())

        # Write to the CSV
        fields = [type_str, n_epochs, neurons, fault_ratio, slices, batch_size_train, learning_rate, loss_val.item(), "notrain"]

        with open(fr'data/{neurons}_4_189_{n_epochs}_sweep_notrainfaults_data.csv', 'a') as f:
            writer = csv.writer(f)
            print(fields)
            writer.writerow(fields)

    print(loss_val_list)


