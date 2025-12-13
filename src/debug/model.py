import pathlib
import sys
import torch

from src.model.rnn import RNN
from src.task.retrocue_wm import RetrocueWMTask

def main():
    dataset = RetrocueWMTask(path="data/retrocue_wm.npy")
    x, y = dataset[:10]
    model = RNN(input_size=x.shape[-1], hidden_size=200, output_size=y.shape[-1])
    out = model(x)

if __name__ == "__main__":
    main()

