from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.draw import disk
from skimage.filters import threshold_otsu
from skimage.measure import label


def synthetic_image(size=128, cells=18, seed=7):
    rng = np.random.default_rng(seed)
    mask = np.zeros((size, size), dtype=bool)
    image = np.zeros((size, size), dtype=float)
    for _ in range(cells):
        radius = int(rng.integers(4, 10))
        center = rng.integers(radius + 1, size - radius - 1, size=2)
        rr, cc = disk(tuple(center), radius, shape=mask.shape)
        mask[rr, cc] = True
        image[rr, cc] += rng.uniform(0.6, 1.0)
    image = gaussian_filter(image, 1.2)
    image += rng.normal(0, 0.12, image.shape)
    image += np.linspace(0, 0.25, size)[:, None]
    return np.clip(image, 0, 1), mask


def segment(image):
    background = gaussian_filter(image, 8)
    corrected = np.clip(image - background, 0, None)
    threshold = threshold_otsu(corrected)
    components = label(corrected > threshold)
    counts = np.bincount(components.ravel())
    keep = counts >= 30
    keep[0] = False
    return keep[components]


def metrics(truth, prediction):
    intersection = np.logical_and(truth, prediction).sum()
    union = np.logical_or(truth, prediction).sum()
    precision = intersection / max(prediction.sum(), 1)
    recall = intersection / max(truth.sum(), 1)
    return {
        "dice": 2 * intersection / max(truth.sum() + prediction.sum(), 1),
        "iou": intersection / max(union, 1),
        "precision": precision,
        "recall": recall,
        "true_objects": int(label(truth).max()),
        "predicted_objects": int(label(prediction).max()),
    }
