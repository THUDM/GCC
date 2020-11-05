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
from tqdm import tqdm

from gcc.datasets.data_util import SSDataset
from gcc.tasks import build_model


class SimilaritySearch(object):
    def __init__(self, dataset_1, dataset_2, model, hidden_size, **model_args):
        """
        Initialize the model.

        Args:
            self: (todo): write your description
            dataset_1: (todo): write your description
            dataset_2: (todo): write your description
            model: (todo): write your description
            hidden_size: (int): write your description
            model_args: (dict): write your description
        """
        self.data = SSDataset("data/panther", dataset_1, dataset_2).data
        self.model = build_model(model, hidden_size, **model_args)
        self.hidden_size = hidden_size

    def _train_wrap(self, data):
        """
        Builds a feature graph from the given data.

        Args:
            self: (todo): write your description
            data: (todo): write your description
        """
        G = nx.MultiGraph()
        G.add_edges_from(data.edge_index.t().tolist())
        embeddings = self.model.train(G)

        # Map node2id
        features_matrix = np.zeros((G.number_of_nodes(), self.hidden_size))
        for vid, node in enumerate(G.nodes()):
            features_matrix[node] = embeddings[vid]
        return features_matrix

    def train(self):
        """
        Train the model.

        Args:
            self: (todo): write your description
        """
        emb_1 = self._train_wrap(self.data[0])
        emb_2 = self._train_wrap(self.data[1])
        return self._evaluate(emb_1, emb_2, self.data[0].y, self.data[1].y)

    def _evaluate(self, emb_1, emb_2, dict_1, dict_2):
        """
        Evaluate the probability of two sets.

        Args:
            self: (todo): write your description
            emb_1: (array): write your description
            emb_2: (array): write your description
            dict_1: (dict): write your description
            dict_2: (dict): write your description
        """
        shared_keys = set(dict_1.keys()) & set(dict_2.keys())
        shared_keys = list(
            filter(
                lambda x: dict_1[x] < emb_1.shape[0] and dict_2[x] < emb_2.shape[0],
                shared_keys,
            )
        )
        emb_1 /= np.linalg.norm(emb_1, axis=1).reshape(-1, 1)
        emb_2 /= np.linalg.norm(emb_2, axis=1).reshape(-1, 1)
        reindex = [dict_2[key] for key in shared_keys]
        reindex_dict = dict([(x, i) for i, x in enumerate(reindex)])
        emb_2 = emb_2[reindex]
        k_list = [20, 40]
        id2name = dict([(dict_2[k], k) for k in dict_2])

        all_results = defaultdict(list)
        for key in shared_keys:
            v = emb_1[dict_1[key]]
            scores = emb_2.dot(v)

            idxs = scores.argsort()[::-1]
            for k in k_list:
                all_results[k].append(int(reindex_dict[dict_2[key]] in idxs[:k]))
        res = dict(
            (f"Recall @ {k}", sum(all_results[k]) / len(all_results[k])) for k in k_list
        )

        return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--emb-path-1", type=str, default="")
    parser.add_argument("--emb-path-2", type=str, default="")
    args = parser.parse_args()
    task = SimilaritySearch(
        args.dataset.split("_")[0],
        args.dataset.split("_")[1],
        args.model,
        args.hidden_size,
        emb_path_1=args.emb_path_1,
        emb_path_2=args.emb_path_2,
    )
    ret = task.train()
    print(ret)
