import os

import torch
import torch.nn as nn

from src.model.rnn import RNN
from src.task.retrocue_wm import RetrocueWMTask, bump_representation
from src.model.loss import AngularLoss, angle_from_bump, angular_difference

def main():
    dataset = RetrocueWMTask(path=f'{os.environ.get("DATA_PATH")}/debug/retrocue_wm.pt')
    x, y = dataset[:10]
    param = {'stimulus': dataset.param['stimulus'][:10], 'location': dataset.param['location'][:10]}
    model = RNN(input_size=x.shape[-1], hidden_size=200, output_size=y.shape[-1])
    out = model(x)

    loss = AngularLoss()
    print(loss(out, y))

    for angle in torch.linspace(-torch.pi, torch.pi, 17):
        bump = bump_representation(angle, 17)
        _angle = angle_from_bump(bump)
        estimated_difference = float(torch.abs(angular_difference(angle, _angle))) 
        real_difference = min([torch.abs(angle-_angle), torch.abs(2*torch.pi+angle-_angle), torch.abs(angle-_angle-2*torch.pi)])
        print(f'Real {angle:.3f} - Predicted {_angle:.3f} - Real Diff {real_difference:.3e} - Estimated Diff {estimated_difference:.3e}')

if __name__ == "__main__":
    main()

