## TODO: переписать нормально _persistent_homologies
## TODO: resamples doc
## TODO: пишем тесты
import numpy as np
import torch
import math
from threading import Thread
from scipy.spatial.distance import cdist
from sklearn.utils.validation import check_array
from sklearn.metrics.pairwise import pairwise_distances_chunked
from sklearn.linear_model import LinearRegression
from _commonfuncs import get_nn, GlobalEstimator


def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
    
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf) ** 0.5


class PH(GlobalEstimator):
    def __init__(self, cuda_device=None, distance_matrix=False, alpha=1.0, restarts=3, resamples=3):
        r"""Intrinsic dimension estimation using the PHDim algorithm.
    
        Parameters
        ----------  
        cuda_device: Optional[str]
            использовать gpu с именем {cuda_device} для подсчета матрицы расстояний.
            Еcли cuda_device is None, тогда подсчет матрицы расстояний идет на cpu
        distance_matrix: bool
            Whether data is a precomputed distance matrix
        alpha: float
            параметр, отвечающий за стоимость ребра. Используется при подсчете MST:
            MST(T) = \sum_{e} |e_i|^{\alpha}
        restarts: int
            number of iterations at each sampling size
        resamples: int
            number of iterations at each sampling size
    
        Attributes
        ----------
        x_: 1d array 
            np.array with the -log(mu) values.
        y_: 1d array 
            np.array with the -log(F(mu_{sigma(i)})) values.
        """        
        self.cuda_device = cuda_device
        self.distance_matrix = distance_matrix
        self.alpha = alpha
        self.restarts = restarts
        self.resamples = resamples
        self.min_points = 40

#        print(self.__sklearn_tags__())

#    def __sklearn_tags__(self):
#        tags = super().__sklearn_tags__()
#        return tags

    def fit(self, X, y=None):
        """A reference implementation of a fitting function.
        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            A data set for which the intrinsic dimension is estimated.
        y : dummy parameter to respect the sklearn API

        Returns
        -------
        self : object
            Returns self.
        """
        X = check_array(X, ensure_min_samples=2, ensure_min_features=2)
        mx_points = X.shape[0]
        step = max(1, ( mx_points - self.min_points ) // 10)
        
        self.dimension_ = self._persistent_homologies(
            X,
            min_points=self.min_points,
            max_points=mx_points,
            point_jump=step,
            restarts=self.restarts,
            resamples=self.resamples
        )

        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

    def _sample_W(self, W, nSamples):
        '''
        Sample <<nSamples>> points from the cloud <<W>>
        '''
        n = W.shape[0]
        random_indices = np.random.choice(n, size=nSamples, replace=False)
        if not self.distance_matrix:
            return W[random_indices]
        return W[random_indices][:, random_indices]

    def _prim_tree(self, adj_matrix, power=1.0):
        '''
        Computation of H0S for a point cloud with distance matrix <<adj_matrix>> by using Prim's algorithm 
        for minimal spanning tree
        '''
        infty = np.max(adj_matrix) + 1.0
    
        dst = np.ones(adj_matrix.shape[0]) * infty
        visited = np.zeros(adj_matrix.shape[0], dtype=bool)
        ancestor = -np.ones(adj_matrix.shape[0], dtype=int)

        v, s = 0, 0.0
        for i in range(adj_matrix.shape[0] - 1):
            visited[v] = 1
            ancestor[dst > adj_matrix[v]] = v
            dst = np.minimum(dst, adj_matrix[v])
            dst[visited] = infty
            
            v = np.argmin(dst)
            
            s += adj_matrix[v][ancestor[v]] ** power
        return s.item()

    def _persistent_homologies(self, W, min_points, max_points, point_jump, alpha=1.0, restarts=3, resamples=3):
        '''
        Estimation of the intrinsic (upper-box) dimension of the given point cloud W.
        Parameters:
        
        min_points --- size of minimal subsample to draw
        max_points --- size of maximal subsample to draw
        point_jump --- size of step between subsamples
        restarts --- number of iterations at each sampling size
        print_error -- to print or not computational error
        '''
        max_points = W.shape[0]

        m_candidates = []
        for i in range(restarts): 
            test_n = range(min_points, max_points, point_jump)
            lengths = []

            for n in test_n:
                reruns = np.ones(resamples)
                for i in range(resamples):
                    tmp = self._sample_W(W, n)
                    if not self.distance_matrix:
                        reruns[i] = self._prim_tree(
                            cdist(tmp, tmp),
                            power=alpha
                        )
                        if self.cuda_device:
                            reruns[i] = self._prim_tree(
                                pairwise_distances(
                                    torch.Tensor(
                                        tmp
                                    ).to(self.cuda_device)
                                ).cpu().detach().numpy(),
                                power=alpha
                            )
                    else:
                        reruns[i] = self._prim_tree(tmp, power=alpha)


                lengths.append(np.median(reruns))

            lengths = np.array(lengths)
            x = np.log(np.array(list(test_n)))
            y = np.log(lengths)

            N = len(x)

            result = (N * (x * y).sum() - x.sum() * y.sum()) / (N * (x ** 2).sum() - x.sum() ** 2)
            if not np.isnan(result):
                m_candidates.append(result)
                
        if len(m_candidates) > 0:
            m = np.mean(m_candidates)
        else:
            m = 0
        return alpha / (1 - m)