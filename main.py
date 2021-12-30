import random
import numpy as np
from sklearn.metrics.cluster import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import yaml
from pyclustertend import hopkins
from argparse import ArgumentParser
from model import MODEL_DICT
from data_loader import prepare_data
import logging

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
    set_seed(args.seed)
    config = read_config(args.config_path)
    model = MODEL_DICT[args.model](**config)
    data = prepare_data(args.data_path, args.imputer)
    hopkins_statistics = hopkins(data, int(0.1 * data.shape[0]))
    logging.info("Hopkins statistics of the dataset: %f" % hopkins_statistics)
    model.fit(data)
    labels = model.predict(data)
    silhouette = silhouette_score(data, labels)
    davies_bouldin = davies_bouldin_score(data, labels)
    calinski_harabasz = calinski_harabasz_score(data, labels)
    logging.info("-----------------Results-----------------")
    logging.info("Silhouette Score is: %f" % silhouette)
    logging.info("Davies Bouldin Score is: %f" % davies_bouldin)
    logging.info("Calinski Harabasz Score is: %f" % calinski_harabasz)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/CC GENERAL.csv')
    parser.add_argument('--config_path', type=str, default='./config/grid.yaml')
    parser.add_argument('--model', type=str, choices=['kmeans', 'grid'], default='grid')
    parser.add_argument('--imputer', type=str, choices=['none', 'simple', 'knn'], default='simple')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args)
