#!/usr/bin/env python
# coding: utf-8

# In[94]:


#get_ipython().system('export PYTHONPATH=$PYTHONPATH:./aihwkit/src/')

# Core imports
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

# Jupyter setup


# # Setting up IBM AI HW Kit

# In[95]:


# Imports from PyTorch.
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
import torch.nn.functional as F
import MultiplyLayer as custom_nn

# Imports from aihwkit.
from aihwkit.nn import AnalogConv1d, AnalogConv2d, AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import SingleRPUConfig, ConstantStepDevice
from aihwkit.simulator.rpu_base import cuda
from aihwkit.simulator.parameters import WeightNoiseType
from aihwkit.simulator.configs import InferenceRPUConfig, WeightModifierType
from aihwkit.simulator.presets import ReRamESPreset
from aihwkit.simulator.presets.devices import ReRamArrayHfO2PresetDevice
from aihwkit.inference import ReRamWan2022NoiseModel, PCMLikeNoiseModel
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator.tiles.base import AnalogTileStateNames, BaseTile, TileModuleBase
from aihwkit.exceptions import TileModuleError


# Create hardware profile to simulate on

# In[96]:


rram_inference_config = InferenceRPUConfig()
#rram_inference_config.mapping.learn_out_scaling = True
#rram_inference_config.mapping.
    # noise_model=PCMLikeNoiseModel())
#rram_inference_config.modifier.pcm_prob_at_random = 0.20
# rram_inference_config.modifier.type = WeightModifierType.PCM_NOISE
# rram_inference_config.modifier.pcm_prob_at_gmax = 0.00 #0.05 
# rram_inference_config.modifier.pcm_prob_at_reset = 0.00

# rram_inference_config.forward.w_noise_type = WeightNoiseType.ADDITIVE_CONSTANT
# rram_inference_config.forward.w_noise = 0.00 #0.01

# rram_inference_config.forward.inp_res = 1/(2^32)
rram_inference_config.forward.out_res
rram_inference_config.forward.out_bound 

from aihwkit.inference.compensation.drift import GlobalDriftCompensation
# from aihwkit.simulator.configs import WeightModifierType
# rram_inference_config.modifier.type = WeightModifierType.ADD_NORMAL
# #rram_inference_config.modifier.std_dev = 0.1
# rram_inference_config.modifier.pdrop = 0.05
rram_inference_config.drift_compensation = GlobalDriftCompensation()



# In[105]:


from aihwkit.inference.compensation.drift import GlobalDriftCompensation
from aihwkit.simulator.parameters.enums import BoundManagementType, NoiseManagementType, WeightClipType, WeightRemapType
from aihwkit.simulator.parameters.io import IOParameters



# Define a single-layer network, using inference/hardware-aware training tile
rpu_config = InferenceRPUConfig()

# rpu_config.pre_post.input_range.learn_input_range = True
# rpu_config.pre_post.input_range.init_from_data = 50
# rpu_config.pre_post.input_range.enable = True
# rpu_config.modifier.pdrop = 0.03  # Drop connect.
# rpu_config.modifier.type = WeightModifierType.ADD_NORMAL  # Fwd/bwd weight noise.
# rpu_config.modifier.std_dev = 0.1
# rpu_config.modifier.rel_to_actual_wmax = True

# Inference noise model.
rpu_config.noise_model = PCMLikeNoiseModel(g_max=25.0)

# drift compensation
rpu_config.drift_compensation = GlobalDriftCompensation()

rpu_config.backward.is_perfect = True


# # Load the dataset
# The file nominal_bundle....pk is created in the previous notebook
# 

# In[106]:


datset_name="data/training_50000_100_[0.001, 0.001, 0.08, 0.08]_[1.0, 1.0, 1.0, 1.0].pk"
with open(datset_name, "rb") as file:
    data = pkl.load(file)
np.random.shuffle(data)


# In[107]:


if torch.cuda.is_available():
    print("PyTorch detected CUDA")
    device = torch.device("cuda:0")
else: 
    print("PyTorch DID NOT detect CUDA")
    device = torch.device("cpu")
print("Training on :", device)


# # Model definition

# In[108]:


# A FFNN with continuous activation functions (softplus)
#128 neurons, one less hidden layer in this version
neurons = 128
model = nn.Sequential(
          #custom_nn.Multiply(1),,
          nn.Linear(6,neurons),
          nn.Softplus(),
          nn.Linear(neurons,neurons),
          nn.Softplus(),  
          nn.Linear(neurons,neurons),
          nn.Softplus(),  
          nn.Linear(neurons,3),
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



#display(model)
print("Amount of weights/parameters:", sum([w.numel() for w in model.parameters() if w.requires_grad]))


# # Convert model to Analog

# In[109]:


model = convert_to_analog(model, rpu_config)
model = model.cuda(device)
print(next(model[0].analog_tiles()).tile)


# # Dataset split and preparation

# In[110]:


# We split the whole dataset into training and testing
X_TRAIN, X_VALIDATION, Y_TRAIN, Y_VALIDATION = train_test_split(data.reshape(-1,10)[:,:6], data.reshape(-1,10)[:,6:9], train_size=0.8)
X_TRAIN = torch.tensor(X_TRAIN, device = device)
X_VALIDATION = torch.tensor(X_VALIDATION, device = device)
Y_TRAIN = torch.tensor(Y_TRAIN, device = device)
Y_VALIDATION = torch.tensor(Y_VALIDATION, device = device)


# # Optimizer and Loss

# In[111]:


# The loss is the cosine of the angle between the predicted and the actual thrust vector
from aihwkit.optim.analog_optimizer import AnalogAdam


cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-8)
# This first loss is directly in degrees
def loss1(ground_truth, prediction):    
    return torch.mean(torch.arccos(cosine_similarity(ground_truth, prediction))) / np.pi * 180
# This is the loss used in the paper with Ekin
def loss2(ground_truth, prediction):    
    return 1 - torch.mean(cosine_similarity(ground_truth, prediction))
loss = loss2

# Setup for the optimizer
learning_rate = 0.5e-4*0.5
optimizer = AnalogAdam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, amsgrad=True)
optimizer.regroup_param_groups(model)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10, verbose=True, threshold=0.005, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

# Lists used to plot loss over epochs
loss_list = []
loss_val_list = []


# # Training loop

# In[112]:


n_epochs = 300 #300 minimum
batch_size_train = 4096
batch_size_val = 4096

first = True
start_time = time.time()

# We store the model that suring training obtains the best score on validation
best_model = copy.deepcopy(model)
best_loss = 1e4

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
        predictions = model(states.float())

        # Loss
        l = loss(labels, predictions)

        # Backward pass = Gradients (dl/dw)
        l.backward() # Automatically compute the gradient of the loss wrt the weights

        # Update weights
        optimizer.step()
        
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
plt.savefig('figures/loss_sim.png')


# In[114]:


# Save model
model_path = f"{datset_name}_{n_epochs}_epochs_4_layers_{neurons}_neurons_{batch_size_train}_batch_size_loss2_analog.mdl"
torch.save(best_model.state_dict(), model_path)


# In[115]:


# Save the loss trend
with open("figures/loss_sim_data.pk","wb") as file:
    pkl.dump((loss_list, loss_val_list), file)

