import numpy as np

from ._graphwave import graphwave_alg


class GraphWave(object):
    def __init__(self, dimension, scale=100, **kwargs):
        self.dimension = dimension
        self.scale = scale

    def train(self, G):
        chi, heat_print, taus = graphwave_alg(
            G, np.linspace(0, self.scale, self.dimension // 4)
        )
        return chi
