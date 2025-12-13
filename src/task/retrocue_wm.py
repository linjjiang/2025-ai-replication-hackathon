import numpy as np
import torch
from torch.utils.data import Dataset
import os

def bump_representation(angle, n_color, kappa = torch.tensor(5.0)):
    phi = torch.linspace(-torch.pi, torch.pi, (n_color+1))[1:]
    return torch.exp(kappa * torch.cos(angle - phi))/(2*torch.pi * torch.special.i0(kappa))

def onehot_representation(index, n, value):
    return torch.zeros(n).scatter_(0, index, value)

class RetrocueWMTask(Dataset):
    def __init__(self, path, n_color = 17, n_location = 2, fixation = 0, stimulus = 1, pre_delay = 7, cue = 1, post_delay = 7):
        self.fixation, self.stimulus, self.pre_delay, self.cue, self.post_delay = fixation, stimulus, pre_delay, cue, post_delay
        self.duration = self.fixation + self.stimulus + self.pre_delay + self.cue + self.post_delay
        if os.path.isfile(path):
            self.x, self.y, self.param = np.load(path)
        else:
            self.param = self.gen_param(n_color, n_location)
            self.x, self.y = self.generate_data(self.param, n_color, n_location)

    def gen_param(self, n_color, n_location):
        param = {}
        angles = torch.linspace(-torch.pi, torch.pi, (n_color + 1))[1:]
        index_grid = torch.cartesian_prod(*([torch.arange(n_color)] * n_location))
        stimulus = angles[index_grid]
        locations = torch.arange(n_location)
        stimulus = stimulus.repeat_interleave(n_location, dim=0)
        param['stimulus'] = stimulus
        param['location'] = locations.repeat(index_grid.size(0))
        return param

    def generate_data(self, param, n_color, n_location):
        n_stimulus = param['stimulus'].shape[0]
        x = torch.zeros((n_stimulus, self.duration, n_location*(n_color+1)))
        y = torch.zeros((n_stimulus, n_color))
        for i in range(n_stimulus):
            stimulus = [bump_representation(param['stimulus'][i,k], n_color) for k in range(n_location)]
            x[i, self.fixation:self.fixation+self.stimulus, n_location:] = torch.cat(stimulus, dim=0)
            x[i, self.fixation+self.stimulus+self.pre_delay:self.fixation+self.stimulus+self.pre_delay+self.cue, param['location'][i]] = 1.0
            y[i] = stimulus[param['location'][i]]  
        return x, y
    
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]