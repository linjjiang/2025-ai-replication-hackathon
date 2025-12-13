import torch
import torch.nn as nn
from torch.nn.functional import softmax

class AngularLoss(nn.Module):
    def __init__(self, beta = 100.0):
        super(AngularLoss, self).__init__()
        self.beta = beta

    def forward(self, out, y):
        weights = torch.linspace(-torch.pi, torch.pi, 1 + out.shape[-1], device=out.device, dtype=out.dtype)[1:]
        angle_out = torch.sum(weights * softmax(self.beta * out, dim=-1), dim=-1)
        angle_y = torch.sum(weights * softmax(self.beta * y, dim=-1), dim=-1)
        diff = angle_out - angle_y
        wrapped = torch.atan2(torch.sin(diff), torch.cos(diff)).unsqueeze(-1)
        return torch.mean(wrapped**2 * (out - y)**2)