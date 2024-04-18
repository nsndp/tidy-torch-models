import cv2
import numpy as np


def face_align(images, landmarks, tform_type, square=True):
    size = images[0].shape[0]
    # 0: left eye, 1: right eye, 2: nose tip, 3: mouth left corner, 4: mouth right corner
    # https://github.com/deepinsight/insightface/issues/1286
    # plus 8 for going from [96, 112] to [112, 112] is included
    lm_dst = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float64)
    if not square:
        lm_dst[:, 0] -= 8
    scale = size / 112
    lm_dst *= scale
    
    for k in range(len(images)):
        if not landmarks[k].any():
            continue
        src = landmarks[k].astype(np.float64)
        #for p in lm[k]: cv2.circle(im, p, 1, (0, 255, 0), 1, cv2.LINE_AA)
        M = estimate_similarity(src, lm_dst) if tform_type == 'similarity' else estimate_affine(src, lm_dst)
        images[k] = cv2.warpAffine(images[k], M, (images[k].shape[1], images[k].shape[0]))
        if not square:
            images[k] = images[k][:, :int(96*scale+1)]
    return images


def estimate_similarity(P1, P2):
    """Estimates similarity transformation by SVD (singular value decomposition) using Umeyama algorithm:
    https://web.stanford.edu/class/cs273/refs/umeyama.pdf (eq. 34-43)
    adapted from: https://github.com/scikit-image/scikit-image/blob/main/skimage/transform/_geometric.py#L91
    equivalent to: import skimage.transform; tform = skimage.transform.SimilarityTransform(); tform.estimate(P1, P2); return tform.params[:2]"""
    n, m = P1.shape # 5, 2
    E1 = np.mean(P1, axis=0)
    E2 = np.mean(P2, axis=0)
    V1 = (P1 - E1).var(axis=0).sum()

    H = (P2 - E2).T @ (P1 - E1) / n
    U, D, VT = np.linalg.svd(H)

    Sc = np.ones(m)
    Sr = np.ones(m)
    if np.linalg.det(H) < 0:
        Sc[-1] = -1
        Sr[-1] = -1
    if np.linalg.matrix_rank(H) == m - 1:
        Sr[-1] = np.sign(np.linalg.det(U) * np.linalg.det(VT))

    R = U @ np.diag(Sr) @ VT
    c = D @ Sc / V1
    t = E2 - c * R @ E1
    return np.hstack([R * c, t[:, None]])


def estimate_affine(P1, P2):
    """Estimates affine transformation by solving a linear matrix equation like this
    [[x1 y1 1]              [[x1n y1n 1]
     [x2 y2 1]  [[a0 b0 0]   [x2n y2n 1]
     [x3 y3 1] @ [a1 b1 0] = [x3n y3n 1]
     [x4 y4 1]   [a2 b2 1]]  [x4n y4n 1]
     [x5 y5 1]]              [x5n y5n 1]]
    adapted from: https://github.com/foamliu/MobileFaceNet/blob/master/align_faces.py#L117"""
    ones = np.ones((P1.shape[0], 1), P1.dtype)
    P1 = np.hstack([P1, ones])
    P2 = np.hstack([P2, ones])
    A, _, _, _ = np.linalg.lstsq(P1, P2, rcond=None)
    return A.T[:2]
    
    
# ==================== NOTES ====================

# One might consider 3 types of transformations to map one set of 2D points to another:
# 1) Projective - 8 DOF (degrees of freedom)
# 2) Affine -     6 DOF - x/y-translation, x/y-rotation, x/y-scale         (special case of projective)
# 3) Similarity - 4 DOF - same but 1 rotation, 1 scale (i.e. no shear) (special case of affine)

# here are some illustrations: https://www.mathworks.com/help/images/ref/fitgeotrans.html#bvonaug
# (in general there are more than 3, of course, but the rest are clearly for different use cases)

# based on my limited tests, projective seems to distort the image way too much to be of real use
# while affine and similarity are about equal (sometimes one is slightly better, sometimes the other)
# affine fits the points better (smaller mean squared error) but warps the image slightly more
# so one might say some useful information (like face proportions) might get lost in these warps
# but at the same time a network will learn easier, with landmarks being in exactly the same place across images
# (and will fail to generalize even harder if you test it on unaligned or differently aligned images,
# but that's the problem with alignment in general)

# In matrix form they look like this:
# (x, y - source point, xn, yn - destination point)
# Similarity
# [[xn]   [[a0 -b0 a1]    [[x]   [[a0*x1 - b0*y1 + a1]   [[s*cos(r)*x - s*sin(r)*y + a1]
#  [yn] =  [b0  a0 b1]  @  [y] =  [b0*x1 + a0*y1 + b1] =  [s*sin(r)*x + s*cos(r)*y + b1]
#  [ 1]]   [ 0   0  1]]    [1]]   [                 1]]   [                           1]]
# Affine
# [[xn]   [[a0 a1 a2]     [[x]   [[a0*x + a1*y + a2]
#  [yn] =  [b0 b1 b2]   @  [y] =  [b0*x + b1*y + b2]
#  [ 1]]   [ 0  0  1]]     [1]]   [               1]]
# Projective
# [[tx]   [[a0 a1 a2]     [[x]   [[a0*x + a1*y + a2]    xn = tx / s
#  [ty] =  [b0 b1 b2]   @  [y] =  [b0*x + b1*y + b2] => yn = ty / s
#  [ s]]   [c0 c1  1]]     [1]]   [c0*x + c1*y + 1]]
# here's an illustration with a matrix for similarity: https://math.stackexchange.com/a/1433328
# skimage docs also has similar helpful descriptions: https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.ProjectiveTransform

# for Umeyama algorithm implementation, I found that some sources calculate scale differently:
# a) by using variance of dst rather than src points: https://github.com/xuexingyu24/MobileFaceNet_Tutorial_Pytorch/blob/master/utils/align_trans.py#L16
#        (same function as in this article:           https://matthewearl.github.io/2015/07/28/switching-eds-with-python/)
# b) by dividing std deviations of both point sets:   https://zpl.fi/aligning-point-patterns-with-kabsch-umeyama-algorithm/
# so instead of:    V1 = (P1 - E1).var(axis=0).sum(); c = D @ Sc / V1
# for a) we have: V2 = np.mean(np.linalg.norm(P2 - E2, axis=1) ** 2); c = V2 / (D @ Sc)
# for b) we have: c = np.std(P2 - E2) / np.std(P1 - E1)
# I'm not good enough at math to say how these differences came to be,
# but in practice both return slightly higher MSE and performs sligtly worse than default (a more than b)
# so it's probably safe to stick to the default at all times

# here's some more about transformations and alignment variations in general:
# https://melgor.github.io/blcv.github.io/static/2017/12/28/demystifying-face-recognition-iii-face-preprocessing/index.html

# some other options I tried:
# cv2.estimateAffine2D(SRC, DST)[0]
#     uses RANSAC algorithm, which is suited for large sets of points with outliers within them, so not really for this use case
# cv2.estimateAffinePartial2D(SRC, DST)[0]
#     same but estimates similarity instead of full affine
# cv2.getAffineTransform(SRC[:3].astype(np.float32), DST[:3].astype(np.float32))
#     gets full affine by solving lin eq system, but expects only 3 points and doesn't accept more, which leads to very imprecise fit
# skimage.transform.ProjectiveTransform().estimate(SRC, DST)
# cv2.findHomography(src_points, dst_points)[0]
# cv2.getPerspectiveTransform(src_points[[0, 1, 3, 4]].astype(np.float32), dst_points[[0, 1, 3, 4]].astype(np.float32))
#     all 3 are projective and distort the image too much

# here's a test function I used for comparison:
def estimate_tform_variants(src_points, dst_points, mode):
    if mode == 'similarity':
        M = estimate_similarity(src_points, dst_points)
    elif mode == 'affine':
        M = estimate_affine(src_points, dst_points)
    elif mode == 'projective':
        import skimage.transform
        tform = skimage.transform.ProjectiveTransform()
        tform.estimate(src_points, dst_points)
        M = tform.params
    #elif ... other options

    # affine and similarity return (2, 3); projective return (3, 3)
    if M.shape[0] == 2:    
        est_points = np.array([M[:, 2] + M[:, :2] @ p for p in src_points])
    else:
        est_points = M @ np.vstack([src_points.T, np.ones(src_points.shape[0])])
        est_points[:2] /= est_points[2]
        est_points = est_points[:2].T
    dist = np.linalg.norm(dst_points - est_points, axis=1)
    mse = np.mean(dist ** 2)
    # later to apply M
    # cv2.warpAffine for (2, 3) [affine/similarity]
    # cv2.warpPerspective for (3, 3) [projective]     
    return M, mse

# and here are the numbers I got on 2 of my test sets:
# (1st number of the overall one-shot classification accuracy using the same model every time (mobilefacenet))
# (2nd number is the mean of all mean squared errors returned by the function above)

# similarity:                       0.9090 / 195.84
# affine:                           0.9076 / 181.96
# similarity (scaling option b):    0.9054 / 206.26 
# similarity (scaling option a):    0.8944 / 243.58 
# cv2.estimateAffinePartial2D:      0.8797 / 351.54
# cv2.estimateAffine2D:             0.8739 / 515.53
# cv2.getPerspectiveTransform:	    0.8079 / 568.48    (without nose)
# skimage.ProjectiveTransform:	    0.7514 / 3778.62 
# cv2.findHomography:	            0.7448 / 243.97
# cv2.getAffineTransform:           0.6092 / 5969.07 (eyes + nose)
# cv2.getAffineTransform:           0.5637 / 4207.67 (mouth + nose)

# affine:                           0.9138 / 52.55
# similarity:                       0.9041 / 58.37
# similarity (scaling option b):    0.9041 / 59.77
# similarity (scaling option a):    0.8944 / 64.47
# cv2.estimateAffinePartial2D:      0.8812 / 104.55
# cv2.estimateAffine2D:             0.8733 / 96.09
# cv2.getPerspectiveTransform:	    0.8698 / 87.13     (w/o nose)
# skimage.ProjectiveTransform:	    0.7326 / 110.44
# cv2.getAffineTransform:           0.6552 / 1558.16 (eyes + nose)