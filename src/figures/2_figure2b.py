import os, sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
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
    activity = torch.load(f'{activity_path}/epoch{epoch:03d}.npy', weights_only=True)

    cued = dataset.param['location']
    stimulus_cued = dataset.param['stimulus'][torch.arange(activity.shape[0]), cued]
    angle_cued = torch.linspace(-torch.pi, torch.pi, dataset.n_color)[stimulus_cued]
    angle_anchor = torch.linspace(-torch.pi, torch.pi, 5)[1:]
    anchor_cued = torch.argmin(torch.abs(angle_cued[:,None] - angle_anchor[None,:]), dim=-1)

    X = torch.zeros((dataset.n_location,)+angle_anchor.shape+activity.shape[1:])
    for location in range(dataset.n_location):
        for i,angle in enumerate(angle_anchor):
            X[location,i] = torch.mean(activity[(cued == location) & (anchor_cued == i)], dim=0)
    X = X.movedim(2,0).detach().numpy()

    pca3d = PCA(n_components=3)
    C = np.zeros(X.shape[:-1]+(3,))
    for i in range(X.shape[0]):
        C[i] = pca3d.fit_transform(X[i].reshape((-1, X.shape[-1]))).reshape(X.shape[1:-1]+(3,))
    
    C_location = np.zeros((C.shape[:-1]+(2,)))
    pca2d = []
    for i in range(X.shape[0]):
        pca2d.append([])
        for location in range(dataset.n_location):
            pca2d[i].append(PCA(n_components=2))
            C_location[i,location] = pca2d[i][location].fit_transform(C[i,location].reshape((-1, C.shape[-1]))).reshape(C.shape[2:-1]+(2,))

    # -------------------------------
    # Display
    # -------------------------------

    i = [7, -1]
    f = plt.figure(figsize=(len(i)*5,4), layout='constrained')
    for j in range(len(i)):
        ax = f.add_subplot(1,len(i),j+1, projection='3d')
        for location, marker in zip(range(dataset.n_location), ['o', '^']):
            ax.scatter(C[i[j],location,:,0], C[i[j],location,:,1], C[i[j],location,:,2], c = angle_anchor, cmap='viridis', alpha = 1.0, marker = marker)
            rx,ry = np.max(np.abs(C_location[i[j],location,:,0])), np.max(np.abs(C_location[i[j],location,:,1]))
            corners_2d = np.array([[-rx, -ry],[ rx, -ry],[ rx,  ry],[-rx,  ry]])
            corners_3d = pca2d[i[j]][location].inverse_transform(corners_2d)
            ax.add_collection3d(Poly3DCollection([corners_3d], facecolors=f'0.8', alpha=0.5, edgecolors='none'))
        ax.set_xlabel(f'PC1 ({pca3d.explained_variance_ratio_[0]:.2f}%)')
        ax.set_ylabel(f'PC2 ({pca3d.explained_variance_ratio_[1]:.2f}%)')
        ax.set_zlabel(f'PC3 ({pca3d.explained_variance_ratio_[2]:.2f}%)')
    if args.show:
        plt.show()
    plt.savefig(figure_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Plotting Figure 2B')
    parser.add_argument('--seed', metavar = 'S', type = int, default = 0, help = 'Seed')
    parser.add_argument('--epoch', metavar = 'E', type = int, default = 299, help = 'Epoch')
    parser.add_argument('--variant', metavar = 'V', type = str, default = '', help = 'Model variant')
    parser.add_argument('--show', action = 'store_true', help = 'Show figure')
    args = parser.parse_args()
    main(args)


