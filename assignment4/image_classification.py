import numpy as np
from skimage import filters
from scipy.spatial.distance import cdist
from sklearn.svm import LinearSVC, SVC
from utils import load_image_gray
import cyvlfeat as vlfeat
import pickle
import math
import multiprocessing as mp


def build_vocabulary(image_paths, vocab_size):
    """
      This function will sample SIFT descriptors from the training images,
      cluster them with kmeans, and then return the cluster centers.

      Useful functions:
      -   Use load_image_gray(path) to load grayscale images
      -   frames, descriptors = vlfeat.sift.dsift(img)
            http://www.vlfeat.org/matlab/vl_dsift.html
              -  frames is a N x 2 matrix of locations, which can be thrown away
              here
              -  descriptors is a N x 128 matrix of SIFT features
            Note: there are step, bin size, and smoothing parameters you can
            manipulate for dsift(). We recommend debugging with the 'fast'
            parameter. This approximate version of SIFT is about 20 times faster to
            compute. Also, be sure not to use the default value of step size. It
            will be very slow and you'll see relatively little performance gain
            from extremely dense sampling. You are welcome to use your own SIFT
            feature code! It will probably be slower, though.
      -   cluster_centers = vlfeat.kmeans.kmeans(X, K)
              http://www.vlfeat.org/matlab/vl_kmeans.html
                -  X is a N x d numpy array of sampled SIFT features, where N is
                   the number of features sampled. N should be pretty large!
                -  K is the number of clusters desired (vocab_size)
                   cluster_centers is a K x d matrix of cluster centers. This is
                   your vocabulary.

      Args:
      -   image_paths: list of image paths.
      -   vocab_size: size of vocabulary

      Returns:
      -   vocab: This is a vocab_size x d numpy array (vocabulary). Each row is a
          cluster center / visual word
    """
    # Load images from the training set. To save computation time, you don't
    # necessarily need to sample from all images, although it would be better
    # to do so. You can randomly sample the descriptors from each image to save
    # memory and speed up the clustering. Or you can simply call vl_dsift with
    # a large step size here, but a smaller step size in get_bags_of_sifts.
    #
    # For each loaded image, get some SIFT features. You don't have to get as
    # many SIFT features as you will in get_bags_of_sift, because you're only
    # trying to get a representative sample here. You can try taking 20 features
    # per image.
    #
    # Once you have tens of thousands of SIFT features from many training
    # images, cluster them with kmeans. The resulting centroids are now your
    # visual word vocabulary.

    # length of the SIFT descriptors that you are going to compute.

    dim = 128
    subsample = 20
#     vocab = np.zeros((vocab_size, dim))
    sift_features = np.zeros((subsample * len(image_paths), dim))
    for idx, p in enumerate(image_paths):
        img_gray = load_image_gray(p)
        _, descriptors = vlfeat.sift.dsift(img_gray, step=16, fast=True)

        indizes = np.random.randint(
            low=0, high=descriptors.shape[0], size=subsample)
        for i, j in enumerate(indizes):
            sift_features[i + subsample * idx] = descriptors[j]

    return vlfeat.kmeans.kmeans(sift_features, vocab_size, algorithm='ELKAN')


def dense_sift(img, vocab):
    hist = np.zeros(vocab.shape[0])
    _, descriptors = vlfeat.sift.dsift(img, step=4, fast=True)
    D = cdist(descriptors, vocab)

    for row in range(D.shape[0]):
        hist[np.argmin(D[row])] += 1

    return hist / np.linalg.norm(hist)


def sift_descriptor(patch):
    """
    Implement a simplifed version of Scale-Invariant Feature Transform (SIFT).
    Paper reference: https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each length of 16/4=4. It is simply the
        terminology used in the feature literature to describe the spatial
        bins where gradient distributions will be described.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length.

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Args:
        patch: grayscale image patch of shape (h, w)

    Returns:
        feature: 1D array of shape (128, )
    """

    dx = filters.sobel_v(patch)
    dy = filters.sobel_h(patch)
    histogram = np.zeros((4, 4, 8))

    h, w = patch.shape[:2]
    for y in range(h // 4):
        for x in range(w // 8):
            dy_w = dy[y * 4: (y + 1) * 4,
                      x * 4: (x + 1) * 4].ravel()
            dx_w = dx[y * 4: (y + 1) * 4,
                      x * 4: (x + 1) * 4].ravel()
            mag_w = np.sqrt(dx_w * dx_w + dy_w * dy_w)
            dirs = ((np.arctan2(dy_w, dx_w) + np.pi * 4)
                    / np.pi).astype(int)
            histogram[y, x] = np.bincount(dirs, weights=mag_w, minlength=8)

    feature = np.reshape(histogram, (128,))
    feature /= np.linalg.norm(feature)
    return feature


def simple_sift_image(img, step=16):
    h, w = img.shape[:2]
    features = np.zeros((h//step * w//step, 128))
    i = 0
    for y in range(0, h, step):
        for x in range(0, w, step):
            if x + 16 >= w or y + 16 >= h:
                continue
            features[i] = sift_descriptor(img[y:y+16, x:x+16])
    return features


def simple_sift_vocab(img, vocab):
    hist = np.zeros(vocab.shape[0])
    descriptors = simple_sift_image(img, step=16)
    D = cdist(descriptors, vocab)

    for row in range(D.shape[0]):
        hist[np.argmin(D[row])] += 1

    return hist / np.linalg.norm(hist)


def build_vocab_simple(image_paths, vocab_size):
    dim = 128
    subsample = 20

    sift_features = np.zeros((subsample * len(image_paths), dim))
    for idx, p in enumerate(image_paths):
        img_gray = load_image_gray(p)
        descriptors = simple_sift_image(img_gray, step=16)

        indizes = np.random.randint(
            low=0, high=descriptors.shape[0], size=subsample)
        for i, j in enumerate(indizes):
            sift_features[i + subsample * idx] = descriptors[j]

    return vlfeat.kmeans.kmeans(sift_features, vocab_size, algorithm='ELKAN')


def bags_of_sifts_simple(image_paths, vocab_filename):
    # load vocabulary
    with open(vocab_filename, 'rb') as f:
        vocab = pickle.load(f)
    print('loaded vocab for simple sift')
    vocab_size = vocab.shape[0]
    # dummy features variable
    feats = np.zeros((len(image_paths), vocab_size))

    for img_idx, p in enumerate(image_paths):
        feats[img_idx] = simple_sift_vocab(load_image_gray(p), vocab)
    return feats


def bags_of_sifts(image_paths, vocab_filename):
    """
      You will want to construct SIFT features here in the same way you
      did in build_vocabulary() (except for possibly changing the sampling
      rate) and then assign each local feature to its nearest cluster center
      and build a histogram indicating how many times each cluster was used.
      Don't forget to normalize the histogram, or else a larger image with more
      SIFT features will look very different from a smaller version of the same
      image.

      Useful functions:
      -   Use load_image(path) to load RGB images and load_image_gray(path) to load
              grayscale images
      -   frames, descriptors = vlfeat.sift.dsift(img)
              http://www.vlfeat.org/matlab/vl_dsift.html
            frames is a M x 2 matrix of locations, which can be thrown away here
            descriptors is a M x 128 matrix of SIFT features
              note: there are step, bin size, and smoothing parameters you can
              manipulate for dsift(). We recommend debugging with the 'fast'
              parameter. This approximate version of SIFT is about 20 times faster
              to compute. Also, be sure not to use the default value of step size.
              It will be very slow and you'll see relatively little performance
              gain from extremely dense sampling. You are welcome to use your own
              SIFT feature code! It will probably be slower, though.
      -   D = cdist(X, Y)
            computes the distance matrix D between all pairs of rows in X and Y.
              -  X is a N x d numpy array of d-dimensional features arranged along
              N rows
              -  Y is a M x d numpy array of d-dimensional features arranged along
              N rows
              -  D is a N x M numpy array where d(i, j) is the distance between row
              i of X and row j of Y

      Args:
      -   image_paths: paths to N images
      -   vocab_filename: Path to the precomputed vocabulary.
              This function assumes that vocab_filename exists and contains an
              vocab_size x 128 ndarray 'vocab' where each row is a kmeans centroid
              or visual word. This ndarray is saved to disk rather than passed in
              as a parameter to avoid recomputing the vocabulary every run.

      Returns:
      -   image_feats: N x d matrix, where d is the dimensionality of the
              feature representation. In this case, d will equal the number of
              clusters or equivalently the number of entries in each image's
              histogram (vocab_size) below.
    """
    # load vocabulary
    with open(vocab_filename, 'rb') as f:
        vocab = pickle.load(f)

    vocab_size = vocab.shape[0]
    # dummy features variable
    feats = np.zeros((len(image_paths), vocab_size))

    for img_idx, p in enumerate(image_paths):
        feats[img_idx] = dense_sift(load_image_gray(p), vocab)
    return feats


def nearest_neighbor_classifier(train_image_feats, train_labels, test_image_feats,
                                metric='euclidean'):
    """
      This function will predict the category for every test image by finding
      the training image with most similar features. Instead of 1 nearest
      neighbor, you can vote based on k nearest neighbors which will increase
      performance (although you need to pick a reasonable value for k).

      Useful functions:
      -   D = cdist(X, Y)
            computes the distance matrix D between all pairs of rows in X and Y.
              -  X is a N x d numpy array of d-dimensional features arranged along
              N rows
              -  Y is a M x d numpy array of d-dimensional features arranged along
              N rows
              -  D is a N x M numpy array where d(i, j) is the distance between row
              i of X and row j of Y

      Args:
      -   train_image_feats:  N x d numpy array, where d is the dimensionality of
              the feature representation
      -   train_labels: N element list, where each entry is a string indicating
              the ground truth category for each training image
      -   test_image_feats: M x d numpy array, where d is the dimensionality of the
              feature representation. You can assume N = M, unless you have changed
              the starter code
      -   metric: (optional) metric to be used for nearest neighbor.
              Can be used to select different distance functions. The default
              metric, 'euclidean' is fine for tiny images. 'chi2' tends to work
              well for histograms

      Returns:
      -   test_labels: M element list, where each entry is a string indicating the
              predicted category for each testing image
    """

    k = 5
    test_labels = [''] * test_image_feats.shape[0]

    D = cdist(test_image_feats, train_image_feats)
    for row in range(D.shape[0]):
        distances = D[row]
        labels = [train_labels[i] for i in np.argpartition(distances, k)[:k]]
        test_labels[row] = max(set(labels), key=labels.count)

    return test_labels


def svm_classify(train_image_feats, train_labels, test_image_feats):
    """
    This function will train a one-versus-one support vector machine (SVM)
    and then use the learned classifiers to predict the category of every test image.

    Useful functions:
    -   sklearn SVC: One-versus-One approach
          https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    -   svm.fit(X, y)

    Args:
    -   train_image_feats:  N x d numpy array, where d is the dimensionality of
            the feature representation
    -   train_labels: N element list, where each entry is a string indicating the
            ground truth category for each training image
    -   test_image_feats: M x d numpy array, where d is the dimensionality of the
            feature representation. You can assume N = M, unless you have changed
            the starter code
    Returns:
    -   test_labels: M element list, where each entry is a string indicating the
            predicted category for each testing image
    """
    categories = list(set(train_labels))
    test_labels = []

    clf = SVC(gamma='scale', decision_function_shape='ovo')
    clf.fit(train_image_feats, train_labels)
    return clf.predict(test_image_feats)


def spatial_pyramid(img, vocab, level, depth):
    if level == depth:
        return np.array([])
    h, w = img.shape[:2]
    weight = 1 / 2**(depth if level == 0 else depth - level + 1)
    return np.concatenate(
        (dense_sift(img, vocab) * weight,
         spatial_pyramid(img[:h//2, :w//2], vocab, level + 1, depth),
         spatial_pyramid(img[:h//2, w//2:], vocab, level + 1, depth),
         spatial_pyramid(img[h//2:, :w//2], vocab, level + 1, depth),
         spatial_pyramid(img[h//2:, w//2:], vocab, level + 1, depth)),
        axis=0)


def bags_of_sifts_spm(image_paths, vocab_filename, depth):
    """
    Bags of sifts with spatial pyramid matching.

    :param image_paths: paths to N images
    :param vocab_filename: Path to the precomputed vocabulary.
          This function assumes that vocab_filename exists and contains an
          vocab_size x 128 ndarray 'vocab' where each row is a kmeans centroid
          or visual word. This ndarray is saved to disk rather than passed in
          as a parameter to avoid recomputing the vocabulary every run.
    :param depth: Depth L of spatial pyramid. Divide images and compute (sum)
          bags-of-sifts for all image partitions for all pyramid levels.
          Refer to the explanation in the notebook, tutorial slide and the
          original paper (Lazebnik et al. 2006.) for more details.

    :return image_feats: N x d matrix, where d is the dimensionality of the
          feature representation. In this case, d will equal the number of
          clusters (vocab_size) times the number of regions in all pyramid levels,
          which is 21 (1+4+16) in this specific case.
    """
    with open(vocab_filename, 'rb') as f:
        vocab = pickle.load(f)

    vocab_size = vocab.shape[0]
    feats = []

    with mp.Pool(mp.cpu_count()) as pool:
        args = [(load_image_gray(p), vocab, 0, depth) for p in image_paths]
        feats = pool.starmap(spatial_pyramid, args)

    feats = np.array(feats)
    print(feats.shape)
    return np.array(feats)
