import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

def main(args):

    # -------------------------------
    # Parameters
    # -------------------------------

    ext = f'_{args.variant}' if args.variant else ''
    model_name = f'RNN_seed{args.seed}{ext}'

    # -------------------------------
    # Paths where to load/save data
    # -------------------------------

	# path where loss are saved
    model_path = f'{os.environ.get("DATA_PATH")}/model/experiment1/{model_name}'
    # path where figures are saved
    figure_path = f'{os.environ.get("FIG_PATH")}/1_figure4a{ext}.png'

    # -------------------------------
    # Prepare data
    # -------------------------------

    train_loss = torch.load(f'{model_path}/train_loss.npy', weights_only=True)

    # -------------------------------
    # Display
    # -------------------------------
    
    f = plt.figure(figsize=(5,2), layout='constrained')
    ax = f.add_subplot(1,1,1)
    ax.plot(train_loss, color='0.0')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    plt.savefig(figure_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Plotting Figure 4A')
    parser.add_argument('--seed', metavar = 'S', type = int, default = 0, help = 'Seed')
    parser.add_argument('--variant', metavar = 'V', type = str, default = '', help = 'Model variant')
    args = parser.parse_args()
    main(args)

