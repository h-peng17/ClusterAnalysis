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
import csv 

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

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
    np.random.seed(seed)


def read_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(args):
    # set_seed(args.seed)
    config = read_config(args.config_path)
    data, _ = prepare_data(args.data_path, args.imputer)
    hopkins_statistics = hopkins(data, int(0.1 * data.shape[0]))
    logging.info("Hopkins statistics of the dataset: %f" % hopkins_statistics)
    silhouette_scores = np.zeros((4, 8), dtype=np.float32)
    davies_bouldin_scores = np.zeros((4, 8), dtype=np.float32)
    calinski_harabasz_scores = np.zeros((4, 8), dtype=np.float32)
    for i, seed in enumerate([42,43,44]):
        set_seed(seed)
        for n_cluters in trange(2, 10):
            model = MODEL_DICT[args.model](n_clusters=n_cluters, **config)
            model.fit(data)
            labels = model.predict(data)
            silhouette = silhouette_score(data, labels)
            davies_bouldin = davies_bouldin_score(data, labels)
            calinski_harabasz = calinski_harabasz_score(data, labels)
            silhouette_scores[i, n_cluters-2] = silhouette
            davies_bouldin_scores[i, n_cluters-2] = davies_bouldin
            calinski_harabasz_scores[i, n_cluters-2] = calinski_harabasz
    silhouette_scores[-1] = np.mean(silhouette_scores[:-1], axis=0)
    davies_bouldin_scores[-1] = np.mean(davies_bouldin_scores[:-1], axis=0)
    calinski_harabasz_scores[-1] = np.mean(calinski_harabasz_scores[:-1], axis=0)
    x = list(range(2, 10))
    plot_lineplot(x, silhouette_scores, "n_cluters", "silhouette_score", f"{args.model}_silhouette_score.png")
    plot_lineplot(x, davies_bouldin_scores, "n_cluters", "davies_bouldin_score", f"{args.model}_davies_bouldin_score.png")
    plot_lineplot(x, calinski_harabasz_scores, "n_cluters", "calinski_harabasz_score", f"{args.model}_calinski_harabasz_score.png")


def main_for_table(args):
    config = read_config(args.config_path)
    data, _ = prepare_data(args.data_path, args.imputer)
    scores = np.zeros((3,5), dtype=np.float32)
    used_time = []
    for i, seed in enumerate([42,43,44]):
        set_seed(seed)
        begin_time = time.time()
        if args.model == "kmeans":
            model = MODEL_DICT[args.model](n_clusters=5, **config)
            model.fit(data)
            labels = model.predict(data)
        elif args.model == "dbscan":
            dbscan = DBSCAN(eps=2, min_samples=10).fit(data)
            labels = dbscan.labels_
        elif args.model == "agglomerative":
            agg = AgglomerativeClustering(n_clusters=5).fit(data)
            labels = agg.labels_
        else:
            raise ValueError
        scores[0][i] = silhouette_score(data, labels)
        scores[1][i] = davies_bouldin_score(data, labels)
        scores[2][i] = calinski_harabasz_score(data, labels)
        used_time.append(time.time()-begin_time)
    for i in range(3):
        scores[i][3] = np.mean(scores[i][:3])
        scores[i][4] = np.std(scores[i][:3])
    used_time = np.sum(np.array(used_time)) / 3
    out_file = "result.txt"
    with open(out_file, "a+") as f:
        result = ""
        for i in range(3):
            for j in range(5):
                result += "&%.4f" % scores[i][j]
        result += "&%.4f" % used_time
        f.write(result + "\n")
    

def visualize(args):
    set_seed(args.seed)
    config = read_config(args.config_path)
    data, _ = prepare_data(args.data_path, args.imputer)
    hopkins_statistics = hopkins(data, int(0.1 * data.shape[0]))
    logging.info("Hopkins statistics of the dataset: %f" % hopkins_statistics)
    model = MODEL_DICT[args.model](n_clusters=5, **config)
    model.fit(data)
    labels = model.predict(data)
    writer.add_embedding(data, labels)


def get_names(input_file):
    with open(input_file, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for idx, line in enumerate(reader):
            if idx == 0:
                return line 


def main_for_interpreter(args):
    set_seed(args.seed)
    config = read_config(args.config_path)
    names = get_names(args.data_path)
    data, scaler = prepare_data(args.data_path, args.imputer)
    model = MODEL_DICT[args.model](n_clusters=5, **config)
    model.fit(data)
    centers = scaler.inverse_transform(model.centroids)
    print(centers.shape)
    result = ""
    for j in range(centers.shape[1]):
        result += names[j] + "&"
        for cluster_id in range(centers.shape[0]):
            result += "%.4f&" % (centers[cluster_id][j])
        result = result[:-1]
        result += "\\\\ \n"
    with open("result.txt", "a+") as f:
        f.write(result)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/CC GENERAL.csv')
    parser.add_argument('--config_path', type=str, default='./config/grid.yaml')
    parser.add_argument('--model', type=str, choices=['kmeans', 'grid', "dbscan", "agglomerative"], default='grid')
    parser.add_argument('--imputer', type=str, choices=['none', 'simple', 'knn'], default='simple')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    # main_for_table(args)
    # main(args)
    visualize(args)
    # main_for_interpreter(args)
