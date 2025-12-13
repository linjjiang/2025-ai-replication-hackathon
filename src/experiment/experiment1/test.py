import os
import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from src.task.retrocue_wm import RetrocueWMTask
from src.model.rnn import RNN
from src.model.generic import GenericModel
from src.model.loss import AngularLoss
import argparse

def main(args):
    
    # -------------------------------
    # Parameters
    # -------------------------------

    pl.seed_everything(args.seed)
    n_epoch = 300
    model_name = f'RNN_seed{args.seed}'
    print(f'Experiment 1 - Testing model {model_name} at epoch {args.epoch[0]}')

    # -------------------------------
    # Paths where to load/save data
    # -------------------------------

    # path where model are saved
    model_path = f'{os.environ.get("DATA_PATH")}/model/experiment1/{model_name}'
    # path where activity saved
    activity_path = f'{os.environ.get("DATA_PATH")}/activity/experiment1/{model_name}'
    os.makedirs(f'{activity_path}', exist_ok=True)

    # -------------------------------
    # Training/Test dataset
    # -------------------------------

    dataset = RetrocueWMTask(path=f'{os.environ.get("DATA_PATH")}/dataset/experiment1.npy')
    loader = DataLoader(dataset, batch_size = 512, shuffle = True, num_workers = 1, pin_memory = True)

    # -------------------------------
    # Loading model
    # -------------------------------

    model = RNN(input_size=dataset.x.shape[-1], hidden_size=200, output_size=dataset.y.shape[-1])
    model = GenericModel(model, loss = AngularLoss())
    model.load_state_dict(torch.load(f'{model_path}/epoch{args.epoch[0]:03d}.ckpt')['state_dict'])
    model.model.return_hidden = True

    # -------------------------------
    # Testing model
    # -------------------------------

    h = []
    for batch in loader:
        x,y = batch
        out, hidden = model(x)
        h.append(hidden)
    h = torch.cat(h, dim=0)
    torch.save(h, f'{activity_path}/epoch{args.epoch[0]:03d}.npy')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Running experiment 1')
    parser.add_argument('--seed', metavar = 'S', type = int, nargs = 1, default = 0, help = 'Seed')
    parser.add_argument('--epoch', metavar = 'E', type = int, nargs = 1, default = 0, help = 'Epoch')
    args = parser.parse_args()
    main(args)