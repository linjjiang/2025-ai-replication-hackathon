
import os

import torch
from src.task.retrocue_wm import RetrocueWMTask

import matplotlib.pyplot as plt

if __name__ == "__main__":
    dataset = RetrocueWMTask(path=f'{os.environ.get("DATA_PATH")}/debug/retrocue_wm.pt', redo = True)
    
    print(dataset.duration)
    print(dataset.param['stimulus'].shape)
    print(dataset.param['location'].shape)
    print(dataset.x.shape)
    print(dataset.y.shape)

    i = 101
    x = dataset.x[i]
    y = torch.full(x.shape[:1]+dataset.y.shape[-1:], float('nan'))
    y[-1] = dataset.y[i]
    data = torch.cat([x, y], dim=1)

    f = plt.figure(figsize=(5,5), layout='constrained')
    ax = f.add_subplot(1,1,1)
    sm = ax.imshow(data.T)
    f.colorbar(sm, cax = ax.inset_axes([1.05, 0, 0.05, 1]))
    ax.set_yticks([-0.5, dataset.n_location-0.5, dataset.n_location+dataset.n_color-0.5, dataset.n_location+dataset.n_color*2-0.5])
    ax.set_yticklabels(['', '', '', ''])
    ax.set_xlabel('Time')
    plt.show()