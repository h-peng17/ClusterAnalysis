import os 
import pdb
import matplotlib
from numpy.random import seed

import torch 


import seaborn as sns 
import matplotlib.pyplot as plt
import pandas as pd 


from pathlib import Path
from collections import Counter
from sklearn.decomposition import PCA

from data_loader import load_data


def plot_lineplot(x, y, x_label, y_label, save_name, out_dir="img"):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    seeds = []
    for i in range(y.shape[0]-1):
        for j in range(y.shape[1]):
            seeds.append(i)
    for j in range(y.shape[1]):
        seeds.append("avg")
    # pdb.set_trace()
    assert len(x*6) == len(seeds)
    assert len(y.reshape(-1)) == len(seeds)
    data = {
        "x": x * 6, 
        "y": y.reshape(-1),
        "seed": seeds
    }
    ax = sns.lineplot(data=data, x="x", y="y", hue="seed")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.savefig(os.path.join(out_dir, save_name), dpi=600, format="png")
    plt.close()


# def plot_lineplot(scores, save_name, out_dir="img"):
#     out_dir = Path(out_dir)
#     out_dir.mkdir(exist_ok=True)
#     data = {
#         "dataset": [],
#         "x": [],
#         "score": []
#     }
#     id2label = {
#         0: "silhouette_score",
#         1: "davies_bouldin_score",
#         2: "calinski_harabasz_score",
#         # 3: "used_time"
#     }
#     for i, _scores in enumerate(scores):
#         for [x, y] in _scores:
#             data["dataset"].append(id2label[i])
#             data["x"].append(x)
#             data["score"].append(y)
#     data = pd.DataFrame(data)
#     # pdb.set_trace()
#     g = sns.FacetGrid(data, col="dataset")
#     g.map(sns.lineplot, "x", "score")
#     # g.set(xticks=[])
#     plt.savefig(save_name, dpi=600, format="png")
#     plt.close()


def analysis_empty(out_dir="img"):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    data = torch.tensor(load_data("./data/CC GENERAL.csv"))
    feature2count = Counter()
    for i in range(data.shape[1]):
        feature2count[i] = data[:, i].isnan().sum().item()
    print(feature2count)
    data = {
        "label": [],
        "count": []
    }
    for key, value in feature2count.items():
        data["label"].append(str(key))
        data["count"].append(value)
    data = pd.DataFrame(data)
    # pdb.set_trace()
    sns.barplot(data=data, x="label", y="count")
    # g.set(xticks=[])
    plt.savefig(os.path.join(out_dir, "empty.png"), dpi=600, format="png")
    plt.close()


def draw(data, centers, out_dir="img"):
    fig, axes = plt.subplots(1,1)
    pca = PCA(n_components=2)
    new_data = pca.fit_transform(data) 
    axes.scatter(new_data[:,0], new_data[:,1],c='b',marker='o',alpha=0.5)
    axes.scatter(centers[:,0], centers[:,1],c='r',marker='*',alpha=0.5)
    plt.savefig(os.path.join(out_dir, "kmeans.png"), dpi=600, format="png")
    plt.close()


if __name__ == "__main__":
    analysis_empty()
