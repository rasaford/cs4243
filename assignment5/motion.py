import numpy as np
import os
from skimage.transform import pyramid_gaussian
from skimage.filters import sobel_h, sobel_v, gaussian
from skimage.feature import corner_harris, corner_peaks


def listdir(path, list_name):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        elif os.path.splitext(file_path)[1] == '.jpg':
            list_name.append(file_path)
    return list_name


def meanShift(dst, track_window, max_iter=100, stop_thresh=1):
    """meanShift tracking method.

    Args:
        dst - Backproject image
        track_window - (x,y,w,h).
        max_iter - the maximum iteration.
        stop_thresh - the stop thresh value.
    Returns:
        track_window - (x,y,w,h).
    """
    completed_iterations = 0
    while True:
        old_mean = track_window[:2]
        x, y, w, h = track_window
        # YOUR CODE HERE
        x_min, x_max = x, min(x+w, dst.shape[1])
        y_min, y_max = y, min(y+h, dst.shape[0])
        center = np.zeros(shape=(2,))
        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                # TODO: use gaussian kernel to improve detection results
                center += np.multiply([x, y], dst[y, x])
        center /= np.sum(dst[y_min:y_max, x_min:x_max])

        center = np.round(center).astype(int)
        top_left = np.maximum(center - np.array([w//2, h//2]), 0)
        # print(dst.shape, top_left, y_min, y_max, x_min,
        #       x_max, dst[y_min:y_max, x_min:x_max].shape)

        # END YOUR CODE
        track_window = np.array([*top_left, w, h])
        if np.linalg.norm(track_window[:2] - old_mean) < stop_thresh or completed_iterations == max_iter:
            print('.', end='', flush=True)
            return track_window
        completed_iterations += 1


def lucas_kanade(img1, img2, keypoints, window_size=9):
    """ Estimate flow vector at each keypoint using Lucas-Kanade method.

    Args:
        img1 - Grayscale image of the current frame. Flow vectors are computed
            with respect to this frame.
        img2 - Grayscale image of the next frame.
        keypoints - Keypoints to track. Numpy array of shape (N, 2).
        window_size - Window size to determine the neighborhood of each keypoint.
            A window is centered around the current keypoint location.
            You may assume that window_size is always an odd number.
    Returns:
        flow_vectors - Estimated flow vectors for keypoints. flow_vectors[i] is
            the flow vector for keypoint[i]. Numpy array of shape (N, 2).

    Hints:
        - You may use np.linalg.inv to compute inverse matrix
    """
    assert window_size % 2 == 1, "window_size must be an odd number"

    flow_vectors = np.zeros(shape=(len(keypoints), 2))
    w = window_size // 2

    # Compute partial derivatives
    Iy, Ix = np.gradient(img1)
    It = img2 - img1

    # For each [y, x] in keypoints, estimate flow vector [vy, vx]
    # using Lucas-Kanade method and append it to flow_vectors.
    for idx, (y, x) in enumerate(keypoints):
        # Keypoints can be loacated between integer pixels (subpixel locations).
        # For simplicity, we round the keypoint coordinates to nearest integer.
        # In order to achieve more accurate results, image brightness at subpixel
        # locations can be computed using bilinear interpolation.
        y, x = int(round(y)), int(round(x))

        A = np.hstack([np.ravel(Ix[y-w:y+w+1, x-w:x+w+1])[np.newaxis].T,
                       np.ravel(Iy[y-w:y+w+1, x-w:x+w+1])[np.newaxis].T])
        b = np.ravel(It[y-w:y+w+1, x-w:x+w+1]) * -1
        x, _ = np.linalg.lstsq(A, b, rcond=None)[0]

        flow_vectors[idx] = x

    return flow_vectors


def iterative_lucas_kanade(img1, img2, keypoints,
                           window_size=9,
                           num_iters=5,
                           g=None):
    """ Estimate flow vector at each keypoint using iterative Lucas-Kanade method.

    Args:
        img1 - Grayscale image of the current frame. Flow vectors are computed
            with respect to this frame.
        img2 - Grayscale image of the next frame.
        keypoints - Keypoints to track. Numpy array of shape (N, 2).
        window_size - Window size to determine the neighborhood of each keypoint.
            A window is centered around the current keypoint location.
            You may assume that window_size is always an odd number.
        num_iters - Number of iterations to update flow vector.
        g - Flow vector guessed from previous pyramid level.
    Returns:
        flow_vectors - Estimated flow vectors for keypoints. flow_vectors[i] is
            the flow vector for keypoint[i]. Numpy array of shape (N, 2).
    """
    assert window_size % 2 == 1, "window_size must be an odd number"

    # Initialize g as zero vector if not provided
    if g is None:
        g = np.zeros(keypoints.shape)

    flow_vectors = []
    w = window_size // 2
    # Compute spatial gradients
    Iy, Ix = np.gradient(img1)

    for y, x, gy, gx in np.hstack((keypoints, g)):
        v = np.zeros(2)  # Initialize flow vector as zero vector
        y1 = int(round(y))
        x1 = int(round(x))

        Ix_w = Ix[y1-w:y1+w+1, x1-w:x1+w+1]
        Iy_w = Iy[y1-w:y1+w+1, x1-w:x1+w+1]

        G = np.array([[np.sum(Ix_w ** 2), np.sum(Ix_w * Iy_w)],
                      [np.sum(Iy_w * Iy_w), np.sum(Iy_w ** 2)]])

        G_inv = np.linalg.inv(G)

        # iteratively update flow vector
        for k in range(num_iters):
            vx, vy = v
            # Refined position of the point in the next frame
            y2 = int(round(y+gy+vy))
            x2 = int(round(x+gx+vx))

            I1 = img1[y1-w:y1+w+1, x1-w:x1+w+1]
            I2 = img2[y2-w:y2+w+1, x2-w:x2+w+1]
            if I1.shape != I2.shape:
                v = np.zeros(2)
                print('Fix this: index out of bounds')
                break

            dIk_w = I1 - I2
            bk = np.array([np.sum(dIk_w * Ix_w),
                           np.sum(dIk_w * Iy_w)])
            v += np.matmul(G_inv, bk)

        flow_vectors.append(v)

    return flow_vectors


def pyramid_lucas_kanade(img1, img2, keypoints,
                         window_size=9, num_iters=5,
                         level=2, scale=2):
    """ Pyramidal Lucas Kanade method

    Args:
        img1 - same as lucas_kanade
        img2 - same as lucas_kanade
        keypoints - same as lucas_kanade
        window_size - same as lucas_kanade
        num_iters - number of iterations to run iterative LK method
        level - Max level in image pyramid. Original image is at level 0 of
            the pyramid.
        scale - scaling factor of image pyramid.

    Returns:
        d - final flow vectors
    """

    # Build image pyramids of img1 and img2
    pyramid1 = tuple(pyramid_gaussian(img1, max_layer=level, downscale=scale))
    pyramid2 = tuple(pyramid_gaussian(img2, max_layer=level, downscale=scale))

    # Initialize pyramidal guess
    g = np.zeros(keypoints.shape)

    for L in range(level, -1, -1):
        # YOUR CODE HERE
        None
        # END YOUR CODE

    d = g + d
    return d


def compute_error(patch1, patch2):
    """ Compute MSE between patch1 and patch2

        - Normalize patch1 and patch2
        - Compute mean square error between patch1 and patch2

    Args:
        patch1 - Grayscale image patch of shape (patch_size, patch_size)
        patch2 - Grayscale image patch of shape (patch_size, patch_size)
    Returns:
        error - Number representing mismatch between patch1 and patch2
    """
    assert patch1.shape == patch2.shape, 'Differnt patch shapes'
    error = 0
    # YOUR CODE HERE

    # END YOUR CODE
    return error


def track_features(frames, keypoints,
                   error_thresh=1.5,
                   optflow_fn=pyramid_lucas_kanade,
                   exclude_border=5,
                   **kwargs):
    """ Track keypoints over multiple frames

    Args:
        frames - List of grayscale images with the same shape.
        keypoints - Keypoints in frames[0] to start tracking. Numpy array of
            shape (N, 2).
        error_thresh - Threshold to determine lost tracks.
        optflow_fn(img1, img2, keypoints, **kwargs) - Optical flow function.
        kwargs - keyword arguments for optflow_fn.

    Returns:
        trajs - A list containing tracked keypoints in each frame. trajs[i]
            is a numpy array of keypoints in frames[i]. The shape of trajs[i]
            is (Ni, 2), where Ni is number of tracked points in frames[i].
    """

    kp_curr = keypoints
    trajs = [kp_curr]
    patch_size = 3  # Take 3x3 patches to compute error
    w = patch_size // 2  # patch_size//2 around a pixel

    for i in range(len(frames) - 1):
        I = frames[i]
        J = frames[i+1]
        flow_vectors = optflow_fn(I, J, kp_curr, **kwargs)
        kp_next = kp_curr + flow_vectors

        new_keypoints = []
        for yi, xi, yj, xj in np.hstack((kp_curr, kp_next)):
            # Declare a keypoint to be 'lost' IF:
            # 1. the keypoint falls outside the image J
            # 2. the error between points in I and J is larger than threshold

            yi = int(round(yi))
            xi = int(round(xi))
            yj = int(round(yj))
            xj = int(round(xj))
            # Point falls outside the image
            if yj > J.shape[0]-exclude_border-1 or yj < exclude_border or\
               xj > J.shape[1]-exclude_border-1 or xj < exclude_border:
                continue

            # Compute error between patches in image I and J
            patchI = I[yi-w:yi+w, xi-w:xi+w]
            patchJ = J[yj-w:yj+w, xj-w:xj+w]
            error = compute_error(patchI, patchJ)
            if error > error_thresh:
                continue

            new_keypoints.append([yj, xj])

        kp_curr = np.array(new_keypoints)
        trajs.append(kp_curr)

    return trajs


def IoU(bbox1, bbox2):
    """ Compute IoU of two bounding boxes

    Args:
        bbox1 - 4-tuple (x, y, w, h) where (x, y) is the top left corner of
            the bounding box, and (w, h) are width and height of the box.
        bbox2 - 4-tuple (x, y, w, h) where (x, y) is the top left corner of
            the bounding box, and (w, h) are width and height of the box.
    Returns:
        score - IoU score
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    score = 0
    # YOUR CODE HERE
    x_left, y_top = max(x1, x2), max(y1, y2)
    x_right, y_bottom = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    isct = (x_right - x_left) * (y_bottom - y_top)
    return float(isct) / (h1 * w1 + h2 * w2 - isct)
