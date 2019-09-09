# -*- coding: utf-8 -*-
from sklearn.utils._joblib import delayed
from sklearn.utils._joblib import Parallel
from sklearn.neighbors import NearestNeighbors
import warnings
from collections import defaultdict
import random
"""
Created on Thu Aug 29 15:34:05 2019

@author: NUS
"""
import numpy as np
#####################################################K-means clustering###########################################
# randomly select the centroids


def randCent(data, k):
    """random gengerate the centroids
    parameters
    ------------
    data: <class 'numpy.ndarray'>, shape=[n_samples, n_features], input data to be randomly select centorids.

    k:    <class 'int'>   the number of the centroids
    ------------
    return
        centroids: <class 'numpy.ndarray'>, shape=[k, n_features]
    """
    return data[np.random.choice(data.shape[0], k), :]


# assuming euclidean norm as the one mentioned by the task description
def dist(x, y):
    return sum([((float(x_1) - float(y_1)))**2 for x_1, y_1 in zip(x, y)])


def KMeans(data, k):
    """ KMeans algorithm
    parameters
    ------------
    data: <class 'numpy.ndarray'>, shape=[n_samples, n_features], input data to be randomly select centorids.

    k:    <class 'int'>   the number of the centroids
    ------------
    return
        centroids: <class 'numpy.ndarray'>, shape=[k, n_features]
        clusterAssment:  <class 'numpy.matrix'>, shape=[n_samples, 1]
    """

    centroids_old = []
    centroids = randCent(data, k)
    clusters = np.zeros(shape=(len(data), 1))
    while True:
        centroids_old = np.copy(centroids)
        for idx, x in enumerate(data):
            clusters[idx] = np.argmin([dist(x, c) for c in centroids])

        for i in range(len(centroids)):
            class_i = np.array([x for idx, x in enumerate(data)
                                if clusters[idx] == i])
            # if not class_i: continue
            centroids[i] = np.sum(class_i, axis=0) / class_i.shape[0]

        if max([dist(centroids[i], centroids_old[i]) for i in range(len(centroids))]) < 1e-2:
            break
    return centroids, clusters


##############################################color #############################################################


def colors(k):
    """ generate the color for the plt.scatter
    parameters
    ------------
    k:    <class 'int'>   the number of the centroids
    ------------
    return
        ret: <class 'list'>, len = k
    """
    ret = []
    for i in range(k):
        ret.append((random.uniform(0, 1), random.uniform(
            0, 1), random.uniform(0, 1)))
    return ret


############################################mean shift clustering##############################################

EPSILON = 1e-2


def _mean_shift_single_seed(my_mean, X, nbrs, max_iter):
    """mean shift cluster for single seed.
    Parameters
    ----------
    X : array-like, shape=[n_samples, n_features]
        Samples to cluster.
    nbrs: NearestNeighbors(radius=bandwidth, n_jobs=1).fit(X)
    max_iter: max interations
    return:
        mean(center) and the total number of pixels which is in the sphere
    """
    # For each seed, climb gradient until convergence or max_iter

    iter = 0
    old_mean = None
    means = np.array([my_mean])
    trace_idx = set()
    while True:
        old_mean = my_mean
        iter += 1

        neigh_idx = nbrs.radius_neighbors([my_mean], return_distance=False)[0]
        neigh = np.array([X[i] for i in neigh_idx])
        my_mean = np.mean(neigh, axis=0, dtype=np.float64)
        means = np.concatenate((means, [my_mean]), axis=0)

        trace_idx |= set(neigh_idx)
        if dist(old_mean, my_mean) < EPSILON or iter >= max_iter:
            return my_mean, trace_idx


def mean_shift(X, bandwidth=None, seeds=None, bin_seeding=False, min_bin_freq=1, cluster_all=True, max_iter=300,
               n_jobs=1):
    """pipline of mean shift clustering
    Parameters
    X : array-like, shape=[n_samples, n_features]
    bandwidth: the radius of the sphere
    seeds: whether use the bin seed algorithm to generate the initial seeds
    bin_size:    bin_size = bandwidth.
    min_bin_freq: for each bin_seed, the minimize of the points should cover
    return:
        cluster_centers <class 'numpy.ndarray'> shape=[n_cluster, n_features] ,labels <class 'list'>, len = n_samples
    """
    # find the points within the sphere
    nbrs = NearestNeighbors(radius=bandwidth, n_jobs=n_jobs).fit(X)
    if not seeds:
        seeds = get_bin_seeds(X, bandwidth, min_bin_freq=min_bin_freq)

    ##########################################parallel computing############################
    all_res = Parallel(n_jobs=n_jobs)(
        delayed(_mean_shift_single_seed)
        (seed, X, nbrs, max_iter) for seed in seeds)
    ##########################################parallel computing############################
    from collections import OrderedDict
    # group nodes
    groups = {tuple(all_res[0][0]): (0, all_res[0][1])}
    key_index = 1
    for res in all_res:
        center = tuple(res[0])
        indices = res[1]
        if min([dist(x, center) for x in groups]) < 1:
            min_key = None
            for x in groups:
                if not min_key or dist(x, center) < dist(min_key, center):
                    min_key = x
            groups[min_key] = (groups[min_key][0],
                               groups[min_key][1] | indices)
        else:
            groups[center] = (key_index, indices)
            key_index += 1

    # assign index of the closest group for each point
    labels = np.array([-1] * len(X))
    curr_group = 0
    for i in range(len(labels)):
        segments = [(k, v[0]) for k, v in groups.items() if i in v[1]]
        labels[i] = segments[np.argmin(
            [dist(pt[0], X[i]) for pt in segments])][1]

    return np.array([x[1] for x in groups]), labels


def get_bin_seeds(X, bin_size, min_bin_freq=1):
    """generate the initial seeds, in order to use the parallel computing
    Parameters
    ----------
    X : array-like, shape=[n_samples, n_features]
    bin_size:    bin_size = bandwidth.
    min_bin_freq: for each bin_seed, the minimize of the points should cover
    return:
        bin_seeds: dict-like bin_seeds = {key=seed, key_value=he total number of pixels which is in the sphere }
    """
    # Bin points
    return np.unique([np.round(x / bin_size) for x in X], axis=0) * bin_size
