from __future__ import absolute_import, division, print_function

import numpy as np
import torch


def extract_image_patch(image, bbox):
    """Extract a BGR image patch from an ``(x, y, width, height)`` box."""
    image = np.asarray(image)
    bbox = np.asarray(bbox, dtype=np.float32).copy()
    bbox[2:] += bbox[:2]

    x1, y1, x2, y2 = bbox.astype(np.int32)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)

    if x1 >= x2 or y1 >= y2:
        return None

    return image[y1:y2, x1:x2]


def to_torch_features(features):
    if isinstance(features, torch.Tensor):
        out = features.detach().float().cpu()
    else:
        out = torch.as_tensor(features, dtype=torch.float32)

    if out.ndim == 1:
        out = out.reshape(1, -1)
    if out.ndim != 2:
        raise ValueError('ReID features must have shape [N, feature_dim].')
    return out


def empty_features(feature_dim):
    return torch.empty((0, int(feature_dim or 0)), dtype=torch.float32)
