import numpy as np


eps = 1e-5


def calc_iou(prediction, ground_truth):
    n_images = len(prediction)
    intersection = np.logical_and(prediction > 0, ground_truth > 0).astype(np.float32).sum((0, 2, 3)) 
    union = np.logical_or(prediction > 0, ground_truth > 0).astype(np.float32).sum((0, 2, 3))
    return intersection, union
