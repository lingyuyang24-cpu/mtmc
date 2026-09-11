"""Deterministic offline ReID association, with no neural-model dependencies.

Track keys are (camera_id, local_id). Inclusive frame spans use each camera's
own timeline; overlapping spans in different cameras are intentionally allowed.
"""

from dataclasses import dataclass, field

import numpy as np


def as_numpy(features):
    """Accept NumPy arrays and detached CPU/GPU torch tensors without importing torch."""
    if hasattr(features, 'detach'):
        features = features.detach().cpu().numpy()
    return np.asarray(features, dtype=np.float64)


def feature_distance(a, b, metric='cosine'):
    """Cosine or ordinary Euclidean distance on L2-normalized embeddings.

    Invalid or zero embeddings have infinite distance under either metric.
    """
    a, b = as_numpy(a), as_numpy(b)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError('Features must be matrices with matching dimensions.')
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    valid_a = np.isfinite(a).all(1) & np.isfinite(a_norm[:, 0]) & (a_norm[:, 0] > 1e-12)
    valid_b = np.isfinite(b).all(1) & np.isfinite(b_norm[:, 0]) & (b_norm[:, 0] > 1e-12)
    a = np.where(valid_a[:, None], a, 0) / np.where(valid_a[:, None], a_norm, 1)
    b = np.where(valid_b[:, None], b, 0) / np.where(valid_b[:, None], b_norm, 1)
    if metric == 'cosine':
        distances = np.clip(1.0 - a @ b.T, 0.0, 2.0)
    elif metric == 'euclidean':
        distances = np.sqrt(np.maximum(
            np.square(a).sum(1)[:, None] + np.square(b).sum(1)[None, :] - 2 * a @ b.T,
            0.0,
        ))
    else:
        raise ValueError('Unknown distance metric: {}'.format(metric))
    distances[~(valid_a[:, None] & valid_b[None, :])] = np.inf
    return distances


@dataclass
class Tracklet:
    key: tuple
    output_id: int
    start: int
    end: int
    features: object
    qualities: object = None


class RepresentativeGallery:
    """Bounded appearance representatives with confidence-weighted support.

Farthest-point selection keeps diverse appearances. Discarded representatives'
weights transfer to their nearest retained representative, including on merge.
"""

    def __init__(self, features, qualities=None, limit=32, metric='cosine'):
        if limit < 1:
            raise ValueError('Gallery size must be positive.')
        if metric not in ('cosine', 'euclidean'):
            raise ValueError('Unknown distance metric: {}'.format(metric))
        features = as_numpy(features)
        if features.size == 0:
            features = np.empty((0, 0)) if features.ndim != 2 else features
        if features.ndim != 2:
            raise ValueError('Features must have shape [N, feature_dim].')
        weights = np.ones(len(features)) if qualities is None else as_numpy(qualities)
        if weights.shape != (len(features),):
            raise ValueError('One quality weight is required per feature.')
        valid = np.isfinite(features).all(axis=1) & np.isfinite(weights) & (weights > 0)
        norms = np.linalg.norm(features, axis=1)
        valid &= np.isfinite(norms) & (norms > 1e-12)
        self.features = features[valid].copy()
        self.weights = weights[valid].copy()
        self.sample_count = len(self.features)
        self.limit, self.metric = limit, metric
        self._bound()

    def _bound(self):
        if len(self.features) <= self.limit:
            return
        selected = [int(np.argmax(self.weights))]
        nearest = feature_distance(self.features, self.features[selected], self.metric)[:, 0]
        while len(selected) < self.limit:
            priority = nearest * self.weights
            priority[selected] = -1
            idx = int(np.argmax(priority))
            if priority[idx] <= 1e-12:
                break
            selected.append(idx)
            nearest = np.minimum(
                nearest, feature_distance(self.features, self.features[[idx]], self.metric)[:, 0]
            )
        distances = feature_distance(self.features, self.features[selected], self.metric)
        assignments = distances.argmin(axis=1)
        self.weights = np.bincount(assignments, weights=self.weights, minlength=len(selected))
        self.features = self.features[selected].copy()

    def distance(self, other):
        if self.metric != other.metric:
            raise ValueError('Cannot compare galleries with different metrics.')
        if not len(self.features) or not len(other.features):
            return float('inf')
        distances = feature_distance(self.features, other.features, self.metric)
        weights = np.outer(self.weights / self.weights.sum(), other.weights / other.weights.sum())
        return float(np.sum(distances * weights))

    def merge(self, other):
        if self.metric != other.metric:
            raise ValueError('Cannot merge galleries with different metrics.')
        if not len(other.features):
            return
        if not len(self.features):
            self.features, self.weights = other.features.copy(), other.weights.copy()
        else:
            self.features = np.concatenate((self.features, other.features))
            self.weights = np.concatenate((self.weights, other.weights))
        self.sample_count += other.sample_count
        self._bound()


@dataclass
class FusionGroup:
    gallery: RepresentativeGallery
    eligible: bool
    members: list = field(default_factory=list)

    def conflicts(self, tracklet):
        return any(
            member.key[0] == tracklet.key[0]
            and max(member.start, tracklet.start) <= min(member.end, tracklet.end)
            for member in self.members
        )


def fuse_tracklets(tracklets, threshold=0.2, margin=0.05, min_frames=10,
                   gallery_size=32, metric='cosine'):
    """Return tuple-key -> global ID and groups, retaining every input tracklet.

Short/invalid tracklets keep their numeric output ID and cannot become merge
targets. A candidate must be compatible with every complete member span, and
beat the second-best compatible group by at least ``margin``. Global IDs are
the stable numeric output IDs of group founders, so ties/order are reproducible.
"""
    if not np.isfinite(threshold) or threshold < 0 or not np.isfinite(margin) or margin < 0:
        raise ValueError('ReID threshold and margin must be finite and nonnegative.')
    if min_frames < 1 or gallery_size < 1:
        raise ValueError('Minimum frames and gallery size must be positive.')
    if metric not in ('cosine', 'euclidean'):
        raise ValueError('Unknown distance metric: {}'.format(metric))
    mapping, groups = {}, {}
    output_ids = set()
    for tracklet in sorted(tracklets, key=lambda item: (item.output_id, item.key)):
        if tracklet.key in mapping or tracklet.output_id in output_ids:
            raise ValueError('Tracklet keys and numeric output IDs must be unique.')
        if tracklet.end < tracklet.start:
            raise ValueError('Tracklet end precedes its start.')
        output_ids.add(tracklet.output_id)
        gallery = RepresentativeGallery(tracklet.features, tracklet.qualities, gallery_size, metric)
        eligible = gallery.sample_count >= min_frames
        candidates = []
        if eligible:
            for global_id, group in groups.items():
                if group.eligible and not group.conflicts(tracklet):
                    candidates.append((gallery.distance(group.gallery), global_id))
        candidates.sort()
        if (candidates and candidates[0][0] < threshold
                and (len(candidates) == 1 or candidates[1][0] - candidates[0][0] >= margin)):
            global_id = candidates[0][1]
            groups[global_id].gallery.merge(gallery)
            groups[global_id].members.append(tracklet)
        else:
            global_id = tracklet.output_id
            groups[global_id] = FusionGroup(gallery, eligible, [tracklet])
        mapping[tracklet.key] = global_id
    return mapping, groups
