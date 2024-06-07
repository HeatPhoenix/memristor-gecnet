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

# Get the current date and time
current_time = datetime.datetime.now()

# Convert the date and time to a string
timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")

# Print the timestamp
print("Run Date Timestamp:", timestamp)

# Store the timestamp in a variable
timestamp_str = str(timestamp)

# Create the parser
parser = argparse.ArgumentParser(description='Train a HW G&CNET')

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
chosen_gpu = args.gpu
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


## Model definition
# A FFNN with continuous activation functions (softplus)
#128 neurons by default, one less hidden layer in this version
if(digital):
    model = nn.Sequential(
        nn.Linear(6, neurons),
        nn.Softplus(),
        nn.Linear(neurons, neurons),
        nn.Softplus(),
        nn.Linear(neurons, neurons),
        nn.Softplus(),
        nn.Linear(neurons, 3),
    )
else:
    model = AnalogSequential(
        AnalogLinearBitSlicing(6, neurons, slices, True, rpu_config=rpu_config),
        nn.Softplus(),
        AnalogLinearBitSlicing(neurons, neurons, slices, True, rpu_config=rpu_config),
        nn.Softplus(),
        AnalogLinearBitSlicing(neurons, neurons, slices, True, rpu_config=rpu_config),
        nn.Softplus(),
        AnalogLinearBitSlicing(neurons, 3, slices, True, rpu_config=rpu_config),
        )
model = model.cuda(device)


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

# Initialize the weights
_ = model.apply(initial_weights)

print("Amount of weights/parameters:", sum([w.numel() for w in model.parameters() if w.requires_grad]))

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

# Setup for the optimizer
if(digital):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
else:
    optimizer = AnalogAdam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, amsgrad=True)
    optimizer.regroup_param_groups(model)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10, verbose=True, threshold=0.005, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

# Lists used to plot loss over epochs
loss_list = []
loss_val_list = []


# # Training loop
batch_size_val = 4096

first = True
start_time = time.time()

# We store the model that suring training obtains the best score on validation
best_model = copy.deepcopy(model)
best_loss = 1e4

scaler = torch.cuda.amp.GradScaler()

print(f'Training batchsize = {batch_size_train}') # Can also be accessed with batchsize = states.shape[0]

for epoch in range(n_epochs):
    if first:
        time_remaining = '-'
    else:
        time_estimate = epoch_time*(n_epochs-epoch+1)
        if time_estimate > 60:
            if time_estimate > 3600:
                time_remaining = str(round(time_estimate/3600,2))+' h'
            else:
                time_remaining = str(round(time_estimate/60,2))+' min'
        else:
            time_remaining = str(round(time_estimate,0))+' s'
        
    first = False
    print(f"Epoch {epoch+1}/{n_epochs}, Current learning rate: {optimizer.state_dict()['param_groups'][0]['lr']}, Time remaining: {'-' if first else time_remaining}")

    start_time_epoch = time.time()
    X_TRAIN, Y_TRAIN = shuffle(X_TRAIN, Y_TRAIN)
    
    # We loop over the entire dataset (ignoring the last incomplete batch)
    for i in tqdm(range(X_TRAIN.shape[0]//batch_size_train)):
        states = X_TRAIN[i*batch_size_train: (i+1)*batch_size_train, :]
        labels = Y_TRAIN[i*batch_size_train: (i+1)*batch_size_train, :]

        # Reset gradients (backward() is cumulative)
        optimizer.zero_grad()
        # Forward pass = prediction
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            predictions = model(states.float())
            # Loss
            l = loss(labels, predictions)


        # Backward pass = Gradients (dl/dw)
        scaler.scale(l).backward() # Automatically compute the gradient of the loss wrt the weights

        # Update weights
        scaler.step(optimizer)
        #optimizer.step()
        scaler.update()
        
    with torch.no_grad():
        # Load random validation batch
        X_VALIDATION, Y_VALIDATION = shuffle(X_TRAIN, Y_TRAIN)
        states_val = X_VALIDATION[:batch_size_val, :]
        labels_val = Y_VALIDATION[:batch_size_val, :]
    
        # Forward pass = prediction
        predictions_val = model(states_val.float())
        
        # Loss
        l_val = loss(labels_val, predictions_val)
        
        if l_val < best_loss:
            best_model = copy.deepcopy(model)
            best_loss = l_val
            print("Updating best model!")
        # Scheduler (reduce learning rate if loss stagnates)
        scheduler.step(l_val)       
    
    # Save values for plots
    loss_list.append(l.item())
    loss_val_list.append(l_val.item())

    print(f'loss = {l:.8f}, loss validation = {l_val:.8f} \n')
    
    epoch_time = (time.time() - start_time_epoch)
    
# Compute excecution time
execution_time = (time.time() - start_time)    
print(f"Total Training Time time: {round(execution_time,2)}s seconds")


# In[113]:


fig, ((ax1)) = plt.subplots(1, 1)
fig.set_figheight(5)
fig.set_figwidth(10)
#ax1.set_title('Loss over epochs')
ax1.grid()
ax1.semilogy(loss_list, 'orange', marker='.',label='Training loss')
ax1.semilogy(loss_val_list, 'red', marker='.', label='Validation loss')
ax1.legend()

type = 'pcm' if pcm else 'rram' if rram else 'digital'
plt.savefig(f'figures/{n_epochs}_epochs_4_layers_{neurons}_neurons_{fault_ratio}_fault_{slices}_slices_{batch_size_train}_batch_size_{type}_loss_sim_{timestamp_str}.png')

# Save model
model_path = f"models/{n_epochs}_epochs_4_layers_{neurons}_neurons_{fault_ratio}_fault_{slices}_slices_{batch_size_train}_batch_size_{type}_loss2_{timestamp_str}.mdl"
torch.save(best_model.state_dict(), model_path)


# Save the loss trend
with open(f"figures/{n_epochs}_epochs_4_layers_{neurons}_neurons_{fault_ratio}_fault_{slices}_slices_{batch_size_train}_batch_size_{type}_loss_sim_{timestamp_str}_data.pk","wb") as file:
    pkl.dump((loss_list, loss_val_list), file)

# Write to the CSV
import csv   
fields=[type, n_epochs, neurons, fault_ratio, slices, batch_size_train, learning_rate, loss_val_list[-1]]
with open(fr'data/{neurons}_4_189_{n_epochs}_sweep_data.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow(fields)