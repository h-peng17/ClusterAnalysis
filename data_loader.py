import os 
import pdb 
import csv
import random

from typing import Dict, Optional, Tuple, List

import torch 
import numpy as np 

from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer, SimpleImputer

def impute(data, _imputer="simple"):
    if _imputer == "simple":
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        data = imputer.fit_transform(data)
    elif _imputer == "knn":
        imputer = KNNImputer(n_neighbors=5)
        data = imputer.fit_transform(data)
    elif _imputer == "none":
        data = torch.tensor(data).nan_to_num()
        pass 
    else:
        raise ValueError
    return data


def standarize(data):
    feature = data[:, :-1]
    label = data[:, -1:]
    norm_feature = StandardScaler().fit_transform(feature)
    data = np.concatenate((norm_feature, label), axis=-1)
    return np.array(data)


def prepare_data(input_file: str, imputer: str="none") -> np.array:
    # load and shuffle
    data = load_data(input_file)
    # np.random.shuffle(data)
    data = standarize(impute(data, _imputer=imputer))
    return data


def load_data(input_file):
    data = []
    with open(input_file, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for idx, line in enumerate(reader):
            if idx == 0:
                for i, item in enumerate(line):
                    print(i, item)
                continue
            data.append([float(x) if x != "" else np.nan for x in line[1:]])
    return np.array(data, dtype=np.float32)


if __name__ == '__main__':
    prepare_data('./data/CC GENERAL.csv')

