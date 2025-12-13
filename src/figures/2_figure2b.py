import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from src.task.retrocue_wm import RetrocueWMTask

def main():

    # -------------------------------
    # Parameters
    # -------------------------------

    model_name = f'RNN_seed0'
    epoch = 249 # 299

    # -------------------------------
    # Paths where to load/save data
    # -------------------------------

	# path where activity are saved
    activity_path = f'{os.environ.get("DATA_PATH")}/activity/experiment1/{model_name}'
    # path where figures are saved
    figure_path = f'{os.environ.get("FIG_PATH")}/2_figure2b.png'

    # -------------------------------
    # Prepare data
    # -------------------------------

    dataset = RetrocueWMTask(path=f'{os.environ.get("DATA_PATH")}/dataset/experiment1.npy')
    activity = torch.load(f'{activity_path}/epoch{epoch:03d}.npy').detach().numpy()
    #print(activity.transpose(0, -1).shape)
    X = activity.reshape((-1, activity.shape[-1]))
    pca = PCA(n_components=3)
    C = pca.fit_transform(X).reshape(activity.shape[:-1]+(3,))
    

    stimulus_cued = dataset.param['stimulus'][torch.arange(C.shape[0]), dataset.param['location']]
    cued = dataset.param['location']

    # -------------------------------
    # Display
    # -------------------------------
    

    i0, i1 = 0, -1
    f = plt.figure(figsize=(6,3), layout='constrained')
    ax = f.add_subplot(1,2,1, projection='3d')
    ax.scatter(C[:,i0,0], C[:,i0,1], C[:,i0,2], c = cued, cmap='viridis')
    ax.set_xlabel('PC1 ({pca.explained_variance_ratio_[0]:.2f}%)')
    ax.set_ylabel('PC2 ({pca.explained_variance_ratio_[1]:.2f}%)')
    ax.set_zlabel('PC3 ({pca.explained_variance_ratio_[2]:.2f}%)')
    ax = f.add_subplot(1,2,2, projection='3d')
    ax.scatter(C[:,i1,0], C[:,i1,1], C[:,i1,2], c = cued, cmap='viridis')
    ax.set_xlabel('PC1 ({pca.explained_variance_ratio_[0]:.2f}%)')
    ax.set_ylabel('PC2 ({pca.explained_variance_ratio_[1]:.2f}%)')
    ax.set_zlabel('PC3 ({pca.explained_variance_ratio_[2]:.2f}%)')
    # plt.show()
    plt.savefig(figure_path)

if __name__ == "__main__":
    main()

