
import os

from src.task.retrocue_wm import RetrocueWMTask

if __name__ == "__main__":
    dataset = RetrocueWMTask(path=f'{os.environ.get("DATA_PATH")}/debug/retrocue_wm.pt')
    
    print(dataset.duration)
    print(dataset.param['stimulus'].shape)
    print(dataset.param['location'].shape)
    print(dataset.x.shape)
    print(dataset.y.shape)