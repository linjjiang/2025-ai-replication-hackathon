import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from src.task.retrocue_wm import RetrocueWMTask

def main(args):

    # -------------------------------
    # Parameters
    # -------------------------------

    ext = f'_{args.variant}' if args.variant else ''
    model_name = f'RNN_seed{args.seed}{ext}'
    epoch = args.epoch

    # -------------------------------
    # Paths where to load/save data
    # -------------------------------

	# path where activity are saved
    activity_path = f'{os.environ.get("DATA_PATH")}/activity/experiment1/{model_name}'
    # path where figures are saved
    figure_path = f'{os.environ.get("FIG_PATH")}/2_figure2b{ext}.png'

    # -------------------------------
    # Prepare data
    # -------------------------------

    dataset = RetrocueWMTask(path=f'{os.environ.get("DATA_PATH")}/dataset/experiment1.pt')
    activity = torch.load(f'{activity_path}/epoch{epoch:03d}.npy', weights_only=True).detach().numpy()
    #print(activity.transpose(0, -1).shape)
    X = activity.reshape((-1, activity.shape[-1]))
    pca = PCA(n_components=3)
    C = pca.fit_transform(X).reshape(activity.shape[:-1]+(3,))
    

    stimulus_cued = dataset.param['stimulus'][torch.arange(C.shape[0]), dataset.param['location']]
    cued = dataset.param['location']

    # -------------------------------
    # Display
    # -------------------------------

    i = [0, 7, 8, -1]
    f = plt.figure(figsize=(len(i)*3,2*3), layout='constrained')
    for j in range(len(i)):
        ax = f.add_subplot(2,len(i),j+1, projection='3d')
        ax.scatter(C[:,i[j],0], C[:,i[j],1], C[:,i[j],2], c = cued, cmap='viridis')
        ax.set_xlabel('PC1 ({pca.explained_variance_ratio_[0]:.2f}%)')
        ax.set_ylabel('PC2 ({pca.explained_variance_ratio_[1]:.2f}%)')
        ax.set_zlabel('PC3 ({pca.explained_variance_ratio_[2]:.2f}%)')
        ax = f.add_subplot(2,len(i),j+1+len(i), projection='3d')
        ax.scatter(C[:,i[j],0], C[:,i[j],1], C[:,i[j],2], c = stimulus_cued, cmap='viridis')
        ax.set_xlabel('PC1 ({pca.explained_variance_ratio_[0]:.2f}%)')
        ax.set_ylabel('PC2 ({pca.explained_variance_ratio_[1]:.2f}%)')
        ax.set_zlabel('PC3 ({pca.explained_variance_ratio_[2]:.2f}%)')
    #plt.show()
    plt.savefig(figure_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Plotting Figure 2B')
    parser.add_argument('--seed', metavar = 'S', type = int, default = 0, help = 'Seed')
    parser.add_argument('--epoch', metavar = 'E', type = int, default = 299, help = 'Epoch')
    parser.add_argument('--variant', metavar = 'V', type = str, default = '', help = 'Model variant')
    args = parser.parse_args()
    main(args)


