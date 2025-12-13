import pathlib
import sys
import torch
import torch.nn as nn

from src.model.rnn import RNN
from src.task.retrocue_wm import RetrocueWMTask
from src.model.loss import AngularLoss

def main():
    dataset = RetrocueWMTask(path="data/retrocue_wm.npy")
    x, y = dataset[:10]
    param = {'stimulus': dataset.param['stimulus'][:10], 'location': dataset.param['location'][:10]}
    model = RNN(input_size=x.shape[-1], hidden_size=200, output_size=y.shape[-1])
    out = model(x)

    mse_loss = nn.MSELoss()
    print(mse_loss(out, y))

    angular_loss = AngularLoss()
    print(angular_loss(out, y))

if __name__ == "__main__":
    main()

