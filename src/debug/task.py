import pathlib
import sys

from src.task.retrocue_wm import RetrocueWMTask

if __name__ == "__main__":
    task = RetrocueWMTask(path="data/retrocue_wm.npy")
    print(task.duration)
    print(task.param['stimulus'].shape)
    print(task.param['location'].shape)
    print(task.x.shape)
    print(task.y.shape)