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

    # -------------------------------
    # Paths where to load/save data
    # -------------------------------

    # path where model are saved
    model_path = f'{os.environ.get("DATA_PATH")}/model/experiment1/{model_name}'
    os.makedirs(f'{model_path}', exist_ok=True)
    # path where log of training are saved
    log_path = f'{os.environ.get("DATA_PATH")}/log/experiment1/{model_name}'
    os.makedirs(f'{log_path}', exist_ok=True)

    # -------------------------------
    # Training/Test dataset
    # -------------------------------

    dataset = RetrocueWMTask(path=f'{os.environ.get("DATA_PATH")}/dataset/experiment1.npy')
    loader = DataLoader(dataset, batch_size = 1, shuffle = True, num_workers = 1, pin_memory = True)

    # -------------------------------
    # Initializing model
    # -------------------------------

    model = RNN(input_size=dataset.x.shape[-1], hidden_size=200, output_size=dataset.y.shape[-1], sigma = 0.07)
    model = GenericModel(model, loss = AngularLoss(), optimizer = 'rmsprop', optimizer_params = {'lr': 0.0001})

    # -------------------------------
    # Saving model
    # -------------------------------

    train_loss = torch.zeros((n_epoch,))

    class MetricsCallback(pl.Callback):
        def on_train_epoch_end(self, *args, **kwargs):
            metrics = trainer.callback_metrics
            train_loss[trainer.current_epoch] = metrics['loss']

    # saving initial model
    torch.save({"epoch": -1, "global_step": 0, "pytorch-lightning_version": pl.__version__, "state_dict": model.state_dict()}, f'{model_path}/epoch{-1:03d}.ckpt')
    # using checkpoint to save models after each epoch
    checkpoint = pl.callbacks.ModelCheckpoint(dirpath=model_path, filename="epoch{epoch:03d}", auto_insert_metric_name=False, save_on_train_epoch_end=True, every_n_epochs=10, save_top_k=-1)
    # print
    metricscb = MetricsCallback()

    # -------------------------------
    # Training model
    # -------------------------------

    trainer = pl.Trainer(default_root_dir=log_path, callbacks=[checkpoint, metricscb], deterministic=True, accelerator="auto", max_epochs=n_epoch)
    trainer.fit(model, loader)
    torch.save(train_loss, f"{model_path}/train_loss.npy")
	

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Running experiment 1')
    parser.add_argument('--seed', metavar = 'S', type = int, nargs = 1, default = 0, help = 'Seed')
    args = parser.parse_args()
    main(args)