import os 
import time 
import random
from matplotlib.pyplot import axis
import numpy as np
from sklearn.metrics.cluster import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import yaml
from pyclustertend import hopkins
from argparse import ArgumentParser
from model import MODEL_DICT
from data_loader import prepare_data
import logging

from sklearn.cluster import KMeans

from torch.utils.tensorboard import SummaryWriter
# os.rmdir("logs/runs")
writer = SummaryWriter("logs/runs")


from tqdm import trange
from analysis import plot_lineplot, draw


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(42)

def read_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main(args):
    # set_seed(args.seed)
    config = read_config(args.config_path)
    data = prepare_data(args.data_path, args.imputer)
    hopkins_statistics = hopkins(data, int(0.1 * data.shape[0]))
    logging.info("Hopkins statistics of the dataset: %f" % hopkins_statistics)
    silhouette_scores = np.zeros((6, 8), dtype=np.float32)
    davies_bouldin_scores = np.zeros((6, 8), dtype=np.float32)
    calinski_harabasz_scores = np.zeros((6, 8), dtype=np.float32)
    # used_time = []
    for i, seed in enumerate([42,43,44,45,46]):
        set_seed(seed)
        for n_cluters in trange(2, 10):
            begin_time = time.time()
            model = MODEL_DICT[args.model](n_clusters=n_cluters, **config)
            model.fit(data)
            labels = model.predict(data)
            silhouette = silhouette_score(data, labels)
            davies_bouldin = davies_bouldin_score(data, labels)
            calinski_harabasz = calinski_harabasz_score(data, labels)
            # logging.info("-----------------Results-----------------")
            # logging.info("Silhouette Score is: %f" % silhouette)
            # logging.info("Davies Bouldin Score is: %f" % davies_bouldin)
            # logging.info("Calinski Harabasz Score is: %f" % calinski_harabasz)
            silhouette_scores[i, n_cluters-2] = silhouette
            davies_bouldin_scores[i, n_cluters-2] = davies_bouldin
            calinski_harabasz_scores[i, n_cluters-2] = calinski_harabasz
            # used_time.append(time.time()-begin_time)
    silhouette_scores[-1] = np.mean(silhouette_scores[:-1], axis=0)
    davies_bouldin_scores[-1] = np.mean(davies_bouldin_scores[:-1], axis=0)
    calinski_harabasz_scores[-1] = np.mean(calinski_harabasz_scores[:-1], axis=0)
    x = list(range(2, 10))
    plot_lineplot(x, silhouette_scores, "n_cluters", "silhouette_score", f"{args.model}_silhouette_score.png")
    plot_lineplot(x, davies_bouldin_scores, "n_cluters", "davies_bouldin_score", f"{args.model}_davies_bouldin_score.png")
    plot_lineplot(x, calinski_harabasz_scores, "n_cluters", "calinski_harabasz_score", f"{args.model}_calinski_harabasz_score.png")
    # plot_lineplot(x, used_time, "n_cluters", "used_time", f"{args.model}_used_time.png")


def visualize(args):
    set_seed(args.seed)
    config = read_config(args.config_path)
    data = prepare_data(args.data_path, args.imputer)
    # model = MODEL_DICT[args.model](n_clusters=5, **config)
    # model.fit(data)
    # labels = model.predict(data)
    kmeans = KMeans(n_clusters=5, random_state=0).fit(data)
    writer.add_embedding(data, kmeans.labels_)
    # draw(data, model.centroids)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/CC GENERAL.csv')
    parser.add_argument('--config_path', type=str, default='./config/grid.yaml')
    parser.add_argument('--model', type=str, choices=['kmeans', 'grid'], default='grid')
    parser.add_argument('--imputer', type=str, choices=['none', 'simple', 'knn'], default='simple')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    # main(args)
    visualize(args)
