# Adapted from udacity deep reinforcement learning examples

import torch
import torch.nn as nn
import torch.nn.functional as F

import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    # Class for handeling the policy network
    
    def __init__(self, state_size, action_size, seed=0, fc_units=None):
        # Initializing the model
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        if fc_units is None:
            self.layers = [nn.Linear(state_size, action_size)]
        else:
            self.layers = [nn.Linear(state_size, fc_units[0])]
            for i, layer_size in enumerate(fc_units[:-1]):
                self.layers.append(nn.Linear(layer_size, fc_units[i+1]))
            self.layers.append(nn.Linear(fc_units[-1], action_size))
        
        self.layers = nn.ModuleList(self.layers)
            
#         for l in self.layers:
#             self.parameters = nn.ParameterList(l.parameters())
    
    def forward(self, state):
        # Building the network to map states -> action values
        for layer in self.layers[:-1]:
            state = F.relu(layer(state))
        
        return self.layers[-1](state)
