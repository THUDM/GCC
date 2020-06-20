import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy import linalg
from scipy.special import iv
from sklearn import preprocessing
from sklearn.utils.extmath import randomized_svd


class ProNE(object):
    def __init__(self, dimension, step=5, mu=0.2, theta=0.5, **kwargs):
        self.dimension = dimension
        self.step = step
        self.mu = mu
        self.theta = theta

    def train(self, G):
        self.num_node = G.number_of_nodes()

        self.matrix0 = sp.csr_matrix(nx.adjacency_matrix(G))

        features_matrix = self._pre_factorization(self.matrix0, self.matrix0)

        embeddings_matrix = self._chebyshev_gaussian(
            self.matrix0, features_matrix, self.step, self.mu, self.theta
        )

        self.embeddings = embeddings_matrix

        return self.embeddings

    def _get_embedding_rand(self, matrix):
        # Sparse randomized tSVD for fast embedding
        l = matrix.shape[0]
        smat = sp.csc_matrix(matrix)  # convert to sparse CSC format
        U, Sigma, VT = randomized_svd(
            smat, n_components=self.dimension, n_iter=5, random_state=None
        )
        U = U * np.sqrt(Sigma)
        U = preprocessing.normalize(U, "l2")
        return U

    def _get_embedding_dense(self, matrix, dimension):
        # get dense embedding via SVD
        U, s, Vh = linalg.svd(
            matrix, full_matrices=False, check_finite=False, overwrite_a=True
        )
        U = np.array(U)
        U = U[:, :dimension]
        s = s[:dimension]
        s = np.sqrt(s)
        U = U * s
        U = preprocessing.normalize(U, "l2")
        return U

    def _pre_factorization(self, tran, mask):
        # Network Embedding as Sparse Matrix Factorization
        l1 = 0.75
        C1 = preprocessing.normalize(tran, "l1")
        neg = np.array(C1.sum(axis=0))[0] ** l1

        neg = neg / neg.sum()

        neg = sp.diags(neg, format="csr")
        neg = mask.dot(neg)

        C1.data[C1.data <= 0] = 1
        neg.data[neg.data <= 0] = 1

        C1.data = np.log(C1.data)
        neg.data = np.log(neg.data)

        C1 -= neg
        F = C1
        features_matrix = self._get_embedding_rand(F)
        return features_matrix

    def _chebyshev_gaussian(self, A, a, order=10, mu=0.5, s=0.5):
        # NE Enhancement via Spectral Propagation
        if order == 1:
            return a

        A = sp.eye(self.num_node) + A
        DA = preprocessing.normalize(A, norm="l1")
        L = sp.eye(self.num_node) - DA

        M = L - mu * sp.eye(self.num_node)

        Lx0 = a
        Lx1 = M.dot(a)
        Lx1 = 0.5 * M.dot(Lx1) - a

        conv = iv(0, s) * Lx0
        conv -= 2 * iv(1, s) * Lx1
        for i in range(2, order):
            Lx2 = M.dot(Lx1)
            Lx2 = (M.dot(Lx2) - 2 * Lx1) - Lx0
            #         Lx2 = 2*L.dot(Lx1) - Lx0
            if i % 2 == 0:
                conv += 2 * iv(i, s) * Lx2
            else:
                conv -= 2 * iv(i, s) * Lx2
            Lx0 = Lx1
            Lx1 = Lx2
            del Lx2
        mm = A.dot(a - conv)
        emb = self._get_embedding_dense(mm, self.dimension)
        return emb
