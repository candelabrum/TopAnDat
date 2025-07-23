import numpy as np
from threading import Thread
from numba import cuda

import math

MINIMAL_CLOUD = 80
USE_64 = True
if USE_64:
    bits = 64
    np_type = np.float64
else:
    bits = 32
    np_type = np.float32

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
    rows = mat.shape[0]
    block_dim = (16, 16)
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

def process_string(sss):
    return sss.replace('\n', ' ').replace('  ', ' ')

class PHD():
    def __init__(self, alpha=1.0, metric='euclidean', n_reruns=3, n_points=7, n_points_min=3):
        self.alpha = alpha
        self.n_reruns = n_reruns
        self.n_points = n_points
        self.n_points_min = n_points_min
        self.metric = metric
        self.is_fitted_ = False
        self.distance_matrix = False

    def _sample_W(self, W, nSamples):
        n = W.shape[0]
        random_indices = np.random.choice(n, size=nSamples, replace=False)
        if self.distance_matrix:
            return W[random_indices][:, random_indices]
        else:
            return W[random_indices]

    def _calc_ph_dim_single(self, W, test_n, outp, thread_id):
        lengths = []
        for n in test_n:
            restarts = self.n_points_min if W.shape[0] <= 2 * n else self.n_points
            reruns = np.ones(restarts)
            for i in range(restarts):
                tmp = self._sample_W(W, n)
                if self.distance_matrix:
                    reruns[i] = prim_tree(tmp, self.alpha)
                else:
                    D = gpu_dist_matrix(tmp)  # GPU-ускоренный cdist
                    reruns[i] = prim_tree(D, self.alpha)
            lengths.append(np.median(reruns))
        x = np.log(np.array(list(test_n)))
        y = np.log(np.array(lengths))
        N = len(x)
        outp[thread_id] = (N * (x * y).sum() - x.sum() * y.sum()) / (N * (x ** 2).sum() - x.sum() ** 2)

    def fit_transform(self, X, y=None, min_points=50, max_points=512, point_jump=40, dist=False):
        self.distance_matrix = dist
        ms = np.zeros(self.n_reruns)
        test_n = range(min_points, max_points, point_jump)
        threads = []
        for i in range(self.n_reruns):
            threads.append(Thread(target=self._calc_ph_dim_single, args=[X, test_n, ms, i]))
            threads[-1].start()
        for i in range(self.n_reruns):
            threads[i].join()
        m = np.mean(ms)
        return 1 / (1 - m)
