# vim: expandtab:ts=4:sw=4
import numpy as np


def non_max_suppression(boxes, max_bbox_overlap, scores=None):
    """Suppress duplicate detections using score-ordered intersection-over-union.

    Original code from [1]_ has been adapted to include confidence score.

    .. [1] http://www.pyimagesearch.com/2015/02/16/
           faster-non-maximum-suppression-python/

    Examples
    --------

        >>> boxes = [d.roi for d in detections]
        >>> scores = [d.confidence for d in detections]
        >>> indices = non_max_suppression(boxes, max_bbox_overlap, scores)
        >>> detections = [detections[i] for i in indices]

    Parameters
    ----------
    boxes : ndarray
        Array of ROIs (x, y, width, height).
    max_bbox_overlap : float
        ROIs that overlap more than this values are suppressed.
    scores : Optional[array_like]
        Detector confidence score.

    Returns
    -------
    List[int]
        Returns indices of detections that have survived non-maxima suppression.

    """
    if len(boxes) == 0:
        return []

    if not 0 <= max_bbox_overlap <= 1:
        raise ValueError('NMS IoU threshold must be between 0 and 1.')
    boxes = np.asarray(boxes, dtype=np.float64)
    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2] + boxes[:, 0]
    y2 = boxes[:, 3] + boxes[:, 1]

    area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    if scores is not None:
        idxs = np.argsort(scores, kind='stable')
    else:
        idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        intersection = w * h
        union = area[i] + area[idxs[:last]] - intersection
        overlap = intersection / np.maximum(union, 1e-12)

        idxs = np.delete(
            idxs, np.concatenate(
                ([last], np.where(overlap > max_bbox_overlap)[0])))

    return pick


def delete_overlap_box(boxes, max_bbox_overlap, scores=None):
    """Compatibility alias; overlapping pairs must never both be deleted."""
    return non_max_suppression(boxes, max_bbox_overlap, scores)


def select_detection_indices(boxes, scores, mode='detector', max_overlap=0.4):
    """Built-in detectors already suppress duplicates; extra NMS is opt-in."""
    if mode == 'detector':
        return list(range(len(boxes)))
    if mode == 'nms':
        return non_max_suppression(boxes, max_overlap, scores)
    raise ValueError('Unknown post-detection NMS mode: {}'.format(mode))
