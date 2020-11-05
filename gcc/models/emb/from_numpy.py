import random

import networkx as nx
import numpy as np


class Zero(object):
    def __init__(self, hidden_size, **kwargs):
        """
        Initialize the hidden.

        Args:
            self: (todo): write your description
            hidden_size: (int): write your description
        """
        self.hidden_size = hidden_size

    def train(self, G):
        """
        Train the graph g.

        Args:
            self: (todo): write your description
            G: (todo): write your description
        """
        return np.zeros((G.number_of_nodes(), self.hidden_size))


class FromNumpy(object):
    def __init__(self, hidden_size, emb_path, **kwargs):
        """
        Initialize hidden data.

        Args:
            self: (todo): write your description
            hidden_size: (int): write your description
            emb_path: (str): write your description
        """
        super(FromNumpy, self).__init__()
        self.hidden_size = hidden_size
        self.emb = np.load(emb_path)

    def train(self, G):
        """
        Train an embedded graph.

        Args:
            self: (todo): write your description
            G: (todo): write your description
        """
        id2node = dict([(vid, node) for vid, node in enumerate(G.nodes())])
        embeddings = np.asarray([self.emb[id2node[i]] for i in range(len(id2node))])
        assert G.number_of_nodes() == embeddings.shape[0]
        return embeddings


class FromNumpyGraph(FromNumpy):
    def train(self, G):
        """
        Train a graph gis graph.

        Args:
            self: (todo): write your description
            G: (array): write your description
        """
        assert G is None
        return self.emb


class FromNumpyAlign(object):
    def __init__(self, hidden_size, emb_path_1, emb_path_2, **kwargs):
        """
        Initialize the target

        Args:
            self: (todo): write your description
            hidden_size: (int): write your description
            emb_path_1: (str): write your description
            emb_path_2: (str): write your description
        """
        self.hidden_size = hidden_size
        self.emb_1 = np.load(emb_path_1)
        self.emb_2 = np.load(emb_path_2)
        self.t1, self.t2 = False, False

    def train(self, G):
        """
        Train the graph.

        Args:
            self: (todo): write your description
            G: (todo): write your description
        """
        if G.number_of_nodes() == self.emb_1.shape[0] and not self.t1:
            emb = self.emb_1
            self.t1 = True
        elif G.number_of_nodes() == self.emb_2.shape[0] and not self.t2:
            emb = self.emb_2
            self.t2 = True
        else:
            raise NotImplementedError

        id2node = dict([(vid, node) for vid, node in enumerate(G.nodes())])
        embeddings = np.asarray([emb[id2node[i]] for i in range(len(id2node))])

        return embeddings
