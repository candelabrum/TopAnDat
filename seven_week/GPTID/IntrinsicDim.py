import numpy as np
import os
from numba import cuda
import math


USE_64 = True
if USE_64:
    bits = 64
    np_type = np.float64
else:
    bits = 32
    np_type = np.float32


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Ограничиваем число потоков BLAS (важно для стабильной загрузки)
# os.environ["OPENBLAS_NUM_THREADS"] = "1"


from scipy.spatial.distance import cdist
from threading import Thread
from line_profiler import profile
from sklearn.metrics import pairwise_distances

MINIMAL_CLOUD = 80

@cuda.jit("void(float{}[:, :], float{}[:, :])".format(bits, bits))
def distance_matrix(mat, out):
    m = mat.shape[0]
    n = mat.shape[1]
    i, j = cuda.grid(2)
    d = 0
    if i < m and j < m:
        for k in range(n):
            tmp = mat[i, k] - mat[j, k]
            d += tmp * tmp
        out[i, j] = math.sqrt(d)

def gpu_dist_matrix(mat):
    # print("I am in gpu")
    rows = mat.shape[0]
    block_dim = (8, 8)
    grid_dim = (int(rows / block_dim[0] + 1), int(rows / block_dim[1] + 1))
    stream = cuda.stream()
    mat2 = cuda.to_device(np.asarray(mat, dtype=np_type), stream=stream)
    out2 = cuda.device_array((rows, rows))
    distance_matrix[grid_dim, block_dim](mat2, out2)
    out = out2.copy_to_host(stream=stream)
    return out

def prim_tree(adj_matrix, alpha=1.0):
    infty = np.max(adj_matrix) + 10
    
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
        s += (adj_matrix[v][ancestor[v]] ** alpha)
        
    return s.item()

import numba


@numba.jit(nopython=True)
def prim_tree_numba(adj_matrix:np.array, alpha=1.0):
    infty = np.max(adj_matrix) + 10
    
    dst = np.ones(adj_matrix.shape[0]) * infty
    visited = np.zeros(adj_matrix.shape[0], dtype=np.bool_)
    ancestor = -np.ones(adj_matrix.shape[0], dtype=np.int_)

    v, s = 0, 0.0
    for i in range(adj_matrix.shape[0] - 1):
        visited[v] = 1
        ancestor[dst > adj_matrix[v]] = v
        dst = np.minimum(dst, adj_matrix[v])
        dst[visited] = infty
        
        v = np.argmin(dst)
        s += (adj_matrix[v][ancestor[v]] ** alpha)
        
    return s.item()

prim_tree_numba(np.array([[0, 1], [1, 0]]))


def process_string(sss):
    return sss.replace('\n', ' ').replace('  ', ' ')

class PHD():
    def __init__(self,
                 alpha=1.0,
                 metric='euclidean',
                 n_reruns=3,
                 n_points=7,
                 n_points_min=3,
                 use_sklearn=False,
                 use_numba=False,
                 use_cuda=False
    ):
        '''
        Initializes the instance of PH-dim computer
        Parameters:
        1) alpha --- real-valued parameter Alpha for computing PH-dim (see the reference paper). Alpha should be chosen lower than the ground-truth Intrinsic Dimensionality; however, Alpha=1.0 works just fine for our kind of data.
        2) metric --- String or Callable, distance function for the metric space (see documentation for Scipy.cdist)
        3) n_reruns --- Number of restarts of whole calculations (each restart is made in a separate thread)
        4) n_points --- Number of subsamples to be drawn at each subsample
        5) n_points_min --- Number of subsamples to be drawn at larger subsamples (more than half of the point cloud)
        '''
        self.alpha = alpha
        self.n_reruns = n_reruns
        self.n_points = n_points
        self.n_points_min = n_points_min
        self.metric = metric
        self.is_fitted_ = False
        self.distance_matrix = False
        self.use_sklearn = use_sklearn
        self.use_numba = use_numba
        self.use_cuda = use_cuda

    def _sample_W(self, W, nSamples):
        n = W.shape[0]
        random_indices = np.random.choice(n, size=nSamples, replace=False)
        if self.distance_matrix:
            return W[random_indices][:, random_indices]
        else:
            return W[random_indices]

    @profile
    def _calc_ph_dim_single(self, W, test_n, outp, thread_id):
        lengths = []
        for n in test_n:
            if W.shape[0] <= 2 * n:
                restarts = self.n_points_min
            else:
                restarts = self.n_points
               
            reruns = np.ones(restarts)
            for i in range(restarts):
                tmp = self._sample_W(W, n)

                if self.distance_matrix:
                    if self.use_numba:
                        reruns[i] = prim_tree_numba(tmp, self.alpha)
                    else:
                        reruns[i] = prim_tree(tmp, self.alpha)
                else:
#                     self.saved_distance_matrix = cdist(tmp, tmp, metric=self.metric)
                    if self.use_sklearn:
                        distance_matrix = pairwise_distances(tmp)
                    elif self.use_cuda:
                        distance_matrix = gpu_dist_matrix(tmp)
                    else:
                        distance_matrix = cdist(tmp, tmp, metric=self.metric)

                    if self.use_numba:
                        reruns[i] = prim_tree_numba(distance_matrix, self.alpha)
                    else:
                        reruns[i] = prim_tree(distance_matrix, self.alpha)

                    
            lengths.append(np.median(reruns))
        lengths = np.array(lengths)

        x = np.log(np.array(list(test_n)))
        y = np.log(lengths)
        N = len(x)   
        outp[thread_id] = (N * (x * y).sum() - x.sum() * y.sum()) / (N * (x ** 2).sum() - x.sum() ** 2)

    def fit_transform(self, X, y=None, min_points=50, max_points=512, point_jump=40, dist=False):
        '''
        Computing the PH-dim 
        Parameters:
        1) X --- point cloud of shape (n_points, n_features), or precomputed distance matrix (n_points, n_points) if parameter dist set to 'True'
        2) y --- fictional parameter to fit with Sklearn interface
        3) min_points --- size of minimal subsample to be drawn
        4) max_points --- size of maximal subsample to be drawn
        5) point_jump --- step between subsamples
        6) dist --- bool value whether X is a precomputed distance matrix
        '''
        self.distance_matrix = dist
        ms = np.zeros(self.n_reruns)
        test_n = range(min_points, max_points, point_jump)
        threads = []

        for i in range(self.n_reruns):
            threads.append(Thread(target=self._calc_ph_dim_single, args=[X, test_n, ms, i]))
            threads[-1].start()

        for i in range(self.n_reruns):
            threads[i].join()

        print(ms)

        m = np.mean(ms)
        return self.alpha / (1 - m)



