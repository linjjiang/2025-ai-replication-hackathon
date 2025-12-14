import torch
import torch.nn as nn
from torch.nn.functional import softmax

def angle_from_bump(out, beta = 100.0):
    weights = torch.linspace(-torch.pi, torch.pi, 1 + out.shape[-1], device=out.device, dtype=out.dtype)[1:]
    return torch.sum(weights * softmax(beta * out, dim=-1), dim=-1)

def angular_difference(angle1, angle2):
    diff = angle1 - angle2
    return torch.atan2(torch.sin(diff), torch.cos(diff)).unsqueeze(-1)

class AngularLoss(nn.Module):
    def __init__(self, beta = 100.0):
        super(AngularLoss, self).__init__()
        self.beta = beta

    def forward(self, out, y):
        angle_out = angle_from_bump(out, beta = self.beta)
        angle_y = angle_from_bump(y, beta = self.beta)
        angle_diff = angular_difference(angle_out, angle_y)
        return torch.mean(angle_diff**2 * (out - y)**2)