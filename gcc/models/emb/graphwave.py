import numpy as np

from ._graphwave import graphwave_alg


class GraphWave(object):
    def __init__(self, dimension, scale=100, **kwargs):
        """
        Initialize a dimension.

        Args:
            self: (todo): write your description
            dimension: (int): write your description
            scale: (float): write your description
        """
        self.dimension = dimension
        self.scale = scale

    def train(self, G):
        """
        Train a graph.

        Args:
            self: (todo): write your description
            G: (array): write your description
        """
        chi, heat_print, taus = graphwave_alg(
            G, np.linspace(0, self.scale, self.dimension // 4)
        )
        return chi
