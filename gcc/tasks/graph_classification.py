import argparse
import copy
import random
import warnings
from collections import defaultdict

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from scipy import sparse as sp
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.utils import shuffle as skshuffle
from tqdm import tqdm

from gcc.datasets.data_util import create_graph_classification_dataset
from gcc.tasks import build_model

warnings.filterwarnings("ignore")


class GraphClassification(object):
    def __init__(self, dataset, model, hidden_size, num_shuffle, seed, **model_args):
        assert model == "from_numpy_graph"
        dataset = create_graph_classification_dataset(dataset)
        self.num_nodes = dataset.graph_labels.shape[0]
        self.num_classes = dataset.num_labels
        self.label_matrix = np.zeros((self.num_nodes, self.num_classes), dtype=int)
        self.labels = dataset.graph_labels

        self.model = build_model(model, hidden_size, **model_args)
        self.hidden_size = hidden_size
        self.num_shuffle = num_shuffle
        self.seed = seed

    def train(self):
        embeddings = self.model.train(None)

        return self.svc_classify(embeddings, self.labels, False)

    def svc_classify(self, x, y, search):
        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.seed)
        accuracies = []
        for train_index, test_index in kf.split(x, y):

            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
            if search:
                params = {"C": [1, 10, 100, 1000, 10000, 100000]}
                classifier = GridSearchCV(
                    SVC(), params, cv=5, scoring="accuracy", verbose=0, n_jobs=-1
                )
            else:
                classifier = SVC(C=100000)
            classifier.fit(x_train, y_train)
            accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))
        return {"Micro-F1": np.mean(accuracies)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-shuffle", type=int, default=10)
    parser.add_argument("--emb-path", type=str, default="")
    args = parser.parse_args()
    task = GraphClassification(
        args.dataset,
        args.model,
        args.hidden_size,
        args.num_shuffle,
        args.seed,
        emb_path=args.emb_path,
    )
    ret = task.train()
    print(ret)
