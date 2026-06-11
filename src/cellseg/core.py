from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.draw import disk, line
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops


CATEGORIES = (
    "cell_body",
    "dendrite",
    "vglut2_puncta",
    "mossy_fiber_rosette",
    "parallel_fiber",
    "climbing_fiber",
)


@dataclass(frozen=True)
class Instance:
    category_id: int
    mask: np.ndarray
    score: float = 1.0


MODEL_CONFIGS = {
    "mask_rcnn": {
        "family": "cnn",
        "backbone": "ResNet-50-FPN",
        "batch_size": 8,
        "learning_rate": 0.02,
        "iterations": 9150,
        "checkpoint_period": 152,
        "workers": 8,
        "confidence_threshold": 0.5,
    },
    "centermask2": {
        "family": "cnn",
        "backbone": "ResNet-50-FPN",
        "batch_size": 8,
        "learning_rate": 0.01,
        "iterations": 9150,
        "confidence_threshold": 0.5,
    },
    "yolact_plus_plus": {
        "family": "cnn",
        "backbone": "ResNet-50",
        "batch_size": 16,
        "iterations": 9150,
        "checkpoint_period": 76,
        "validation_period_epochs": 2,
    },
    "mask2former": {
        "family": "transformer",
        "backbone": "ResNet-50",
        "batch_size": 6,
        "learning_rate": 0.0001,
        "iterations": 12200,
        "checkpoint_period": 200,
    },
    "maskdino": {
        "family": "transformer",
        "backbone": "ResNet-50",
        "batch_size": 8,
        "learning_rate": 0.0001,
        "iterations": 9150,
        "checkpoint_period": 200,
        "confidence_threshold": 0.5,
    },
    "fastinst": {
        "family": "transformer",
        "batch_size": 8,
        "learning_rate": 0.0001,
        "iterations": 9150,
        "checkpoint_period": 200,
        "evaluation_iteration": 4575,
    },
}

PUBLISHED_RESULTS = {
    "mask_rcnn": [23.3, 43.0, 23.7, 2.0, 16.0, 29.5, 12.9, 28.7, 30.9, 5.5, 23.1, 34.3],
    "centermask2": [25.2, 46.0, 25.0, 9.1, 20.6, 32.4, 14.5, 31.0, 33.1, 17.2, 30.9, 36.3],
    "yolact_plus_plus": [17.9, 33.9, 16.0, 3.9, 14.3, 26.9, 11.9, 23.2, 24.3, 10.2, 26.3, 31.1],
    "mask2former": [30.2, 48.6, 30.1, 1.0, 13.4, 41.3, 17.6, 34.1, 38.7, 5.2, 24.5, 48.2],
    "maskdino": [29.2, 51.7, 28.7, 2.3, 15.2, 38.8, 17.2, 35.1, 38.4, 6.0, 23.9, 47.0],
    "fastinst": [21.0, 36.6, 18.9, 0.8, 11.6, 27.3, 16.4, 32.2, 35.0, 3.1, 17.6, 43.2],
}
PUBLISHED_COLUMNS = (
    "AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR75_100", "ARs", "ARm", "ARl"
)


def synthetic_image(size: int = 192, seed: int = 7) -> tuple[np.ndarray, list[Instance]]:
    rng = np.random.default_rng(seed)
    image = np.zeros((size, size, 3), dtype=float)
    instances: list[Instance] = []
    for category_id in range(1, 7):
        for _ in range(2 + category_id % 2):
            mask = np.zeros((size, size), dtype=bool)
            center = rng.integers(20, size - 20, size=2)
            if category_id in (1, 3, 4):
                radius = int(rng.integers(3, 12))
                rr, cc = disk(tuple(center), radius, shape=mask.shape)
            else:
                end = np.clip(center + rng.integers(-35, 36, size=2), 2, size - 3)
                rr, cc = line(*center, *end)
                for offset in (-1, 0, 1):
                    mask[np.clip(rr + offset, 0, size - 1), cc] = True
                rr, cc = np.where(mask)
            mask[rr, cc] = True
            channel = (category_id - 1) % 3
            image[..., channel] += gaussian_filter(mask.astype(float), 1.0) * rng.uniform(0.7, 1)
            instances.append(Instance(category_id, mask))
    image += rng.normal(0, 0.05, image.shape)
    return np.clip(image, 0, 1), instances


def mask_bbox(mask: np.ndarray) -> list[float]:
    rows, columns = np.where(mask)
    return [
        float(columns.min()),
        float(rows.min()),
        float(columns.max() - columns.min() + 1),
        float(rows.max() - rows.min() + 1),
    ]


def coco_record(image_id: int, image: np.ndarray, instances: list[Instance]) -> dict:
    annotations = []
    for annotation_id, instance in enumerate(instances, start=1):
        annotations.append(
            {
                "id": image_id * 1000 + annotation_id,
                "image_id": image_id,
                "category_id": instance.category_id,
                "bbox": mask_bbox(instance.mask),
                "area": int(instance.mask.sum()),
                "iscrowd": 0,
            }
        )
    return {
        "images": [{"id": image_id, "width": image.shape[1], "height": image.shape[0]}],
        "annotations": annotations,
        "categories": [{"id": index + 1, "name": name} for index, name in enumerate(CATEGORIES)],
    }


def reference_segment(image: np.ndarray) -> list[Instance]:
    predictions = []
    for channel in range(3):
        corrected = np.clip(image[..., channel] - gaussian_filter(image[..., channel], 8), 0, None)
        components = label(corrected > threshold_otsu(corrected))
        for region in regionprops(components):
            if region.area < 8:
                continue
            mask = components == region.label
            predictions.append(Instance(channel + 1, mask, score=min(0.99, 0.5 + region.area / 500)))
    return predictions


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = np.logical_and(left, right).sum()
    union = np.logical_or(left, right).sum()
    return float(intersection / max(union, 1))


def match_instances(
    truth: list[Instance], prediction: list[Instance], threshold: float
) -> tuple[int, int, int]:
    used: set[int] = set()
    true_positive = 0
    for candidate in sorted(prediction, key=lambda item: item.score, reverse=True):
        matches = [
            (mask_iou(candidate.mask, target.mask), index)
            for index, target in enumerate(truth)
            if index not in used and candidate.category_id == target.category_id
        ]
        if matches and max(matches)[0] >= threshold:
            used.add(max(matches)[1])
            true_positive += 1
    return true_positive, len(prediction) - true_positive, len(truth) - true_positive


def evaluate_instances(truth: list[Instance], prediction: list[Instance]) -> dict[str, float]:
    metrics = {}
    precisions, recalls = [], []
    for threshold in np.arange(0.50, 1.00, 0.05):
        tp, fp, fn = match_instances(truth, prediction, float(threshold))
        precisions.append(tp / max(tp + fp, 1))
        recalls.append(tp / max(tp + fn, 1))
        if np.isclose(threshold, 0.50):
            metrics["AP50"] = precisions[-1] * 100
        if np.isclose(threshold, 0.75):
            metrics["AP75"] = precisions[-1] * 100
            metrics["AR75_100"] = recalls[-1] * 100
    metrics["AP"] = float(np.mean(precisions) * 100)
    metrics["AR100"] = float(np.mean(recalls) * 100)
    return metrics
