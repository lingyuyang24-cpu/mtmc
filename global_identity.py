"""Online identity association, independent of video capture and neural models.

Only confirmed, appearance-consistent observations may update a global gallery.
Use update_camera() to reserve identities for all people in a camera frame.
"""

from collections import deque
import math

import numpy as np


def normalize_feature(feature):
    feature = np.asarray(feature, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(feature)
    if not feature.size or not np.all(np.isfinite(feature)) or norm <= 1e-12:
        return None
    return feature / norm


def mean_feature(features, qualities=None):
    features = list(features)
    weights = [1.0] * len(features) if qualities is None else list(qualities)
    if len(weights) != len(features):
        raise ValueError('Features and quality weights must have the same length.')
    valid, valid_weights = [], []
    for feature, weight in zip(features, weights):
        feature = normalize_feature(feature)
        if feature is not None and np.isfinite(weight) and weight > 0:
            valid.append(feature)
            valid_weights.append(weight)
    if not valid:
        return None
    return normalize_feature(np.average(valid, axis=0, weights=valid_weights))


def cosine_distance(query, gallery):
    query = normalize_feature(query)
    gallery = [normalize_feature(feature) for feature in gallery]
    gallery = [feature for feature in gallery if feature is not None]
    if query is None or not gallery:
        return None
    return np.clip(1.0 - np.dot(np.asarray(gallery), query), 0.0, 2.0)


def euclidean_distance(query, gallery):
    # Both association and gallery updates use the selected metric on unit vectors.
    distances = cosine_distance(query, gallery)
    return None if distances is None else np.sqrt(2.0 * distances)


def bbox_center_distance_ratio(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    distance = math.hypot((ax1 + ax2 - bx1 - bx2) / 2,
                          (ay1 + ay2 - by1 - by2) / 2)
    return distance / max(1.0, math.hypot(ax2 - ax1, ay2 - ay1),
                          math.hypot(bx2 - bx1, by2 - by1))


def bbox_iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return intersection / max(1e-12, area_a + area_b - intersection)


class LocalTrackState:
    def __init__(self, camera_id, local_id, now, max_features, aggregate_frames):
        self.camera_id = int(camera_id)
        self.local_id = int(local_id)
        self.first_seen = self.last_seen = float(now)
        self.last_bbox = None
        self.features = deque(maxlen=max_features)
        self.feature_qualities = deque(maxlen=max_features)
        self.pending_features = deque(maxlen=aggregate_frames)
        self.pending_qualities = deque(maxlen=aggregate_frames)
        self.global_id = None
        self.last_global_update = 0.0
        self.candidate_id = None
        self.candidate_source = None
        self.candidate_votes = 0
        self.candidate_started = None
        self.candidate_progress = None
        self.assignment_started = None
        self.reconnect_features = []
        self.reconnect_qualities = []
        self.reconnect_matches = []
        self.mismatch_windows = 0
        self.suspect_features = []
        self.suspect_qualities = []
        self.blocked_ids = set()

    def add_observation(self, feature, bbox, quality, now):
        self.last_seen = float(now)
        if bbox is not None:
            self.last_bbox = tuple(float(v) for v in bbox)
        if feature is None or not np.isfinite(quality) or quality <= 0:
            return None
        feature = normalize_feature(feature)
        if feature is None:
            return None
        self.features.append(feature)
        self.feature_qualities.append(float(quality))
        self.pending_features.append(feature)
        self.pending_qualities.append(float(quality))
        return feature

    def mean_quality(self):
        return float(np.mean(self.feature_qualities)) if self.feature_qualities else 0.0

    def clear_pending_features(self):
        self.pending_features.clear()
        self.pending_qualities.clear()

    def pop_aggregated_feature(self, min_features):
        if len(self.pending_features) < min_features:
            return None, 0.0
        feature = mean_feature(self.pending_features, self.pending_qualities)
        quality = float(np.mean(self.pending_qualities))
        self.clear_pending_features()
        return feature, quality

    def clear_candidate(self):
        self.candidate_id = self.candidate_source = self.candidate_started = None
        self.candidate_progress = None
        self.candidate_votes = 0
        self.reconnect_features = []
        self.reconnect_qualities = []
        self.reconnect_matches = []


class GlobalIdentity:
    def __init__(self, global_id, gallery_size, anchor_size, now):
        self.global_id = int(global_id)
        self.anchor_size = anchor_size
        self.anchor_features = []
        self.recent_features = deque(maxlen=gallery_size - anchor_size)
        self.created_at = self.last_seen = float(now)
        self.camera_states = {}

    @property
    def features(self):
        return self.anchor_features + list(self.recent_features)


class GlobalIDManager:
    def __init__(
        self, threshold=0.3, margin=0.02, delay_seconds=1.0, min_features=5,
        gallery_size=80, track_feature_size=40, anchor_size=20,
        anchor_min_confidence=0.65, aggregate_frames=5,
        feature_min_novelty=0.03, feature_update_max_distance=0.35,
        metric='cosine', topk=3, feature_update_interval=0.2,
        active_timeout=0.3, stale_timeout=1800.0, same_camera_reconnect=True,
        same_camera_reconnect_timeout=5.0, same_camera_reconnect_distance=0.4,
        same_camera_reconnect_min_iou=0.02, reconnect_confirm_frames=10,
        reconnect_confirm_ratio=0.6, reconnect_confirm_threshold=0.35,
        confirm_windows=3, revalidate_threshold=0.4, revalidate_windows=3,
        pending_timeout=10.0
    ):
        if metric not in ('cosine', 'euclidean'):
            raise ValueError('Unknown ReID metric: {}'.format(metric))
        if not 1 <= anchor_size <= gallery_size:
            raise ValueError('anchor_size must be between 1 and gallery_size.')
        if min(min_features, aggregate_frames, confirm_windows, revalidate_windows,
               reconnect_confirm_frames, topk) < 1:
            raise ValueError('Feature, confirmation and top-k counts must be positive.')
        if track_feature_size < max(min_features, aggregate_frames):
            raise ValueError('track_feature_size must cover min_features and aggregate_frames.')
        if min(threshold, revalidate_threshold, reconnect_confirm_threshold,
               feature_update_max_distance, pending_timeout) <= 0:
            raise ValueError('Distance thresholds and pending_timeout must be positive.')
        if min(delay_seconds, margin, active_timeout, stale_timeout,
               same_camera_reconnect_timeout, feature_update_interval,
               feature_min_novelty, same_camera_reconnect_distance) < 0:
            raise ValueError('Timeouts, margins and distances must be nonnegative.')
        if not 0 < reconnect_confirm_ratio <= 1:
            raise ValueError('reconnect_confirm_ratio must be in (0, 1].')
        if not 0 <= same_camera_reconnect_min_iou <= 1:
            raise ValueError('same_camera_reconnect_min_iou must be in [0, 1].')
        if not 0 <= anchor_min_confidence <= 1:
            raise ValueError('anchor_min_confidence must be in [0, 1].')
        if feature_min_novelty > feature_update_max_distance:
            raise ValueError('Feature novelty threshold exceeds update distance.')
        self.threshold = threshold
        self.margin = margin
        self.delay_seconds = delay_seconds
        self.min_features = min_features
        self.gallery_size = gallery_size
        self.track_feature_size = track_feature_size
        self.anchor_size = anchor_size
        self.anchor_min_confidence = anchor_min_confidence
        self.aggregate_frames = aggregate_frames
        self.feature_min_novelty = feature_min_novelty
        self.feature_update_max_distance = feature_update_max_distance
        self.metric = metric
        self.topk = topk
        self.feature_update_interval = feature_update_interval
        self.active_timeout = active_timeout
        self.stale_timeout = stale_timeout
        self.same_camera_reconnect = same_camera_reconnect
        self.same_camera_reconnect_timeout = same_camera_reconnect_timeout
        self.same_camera_reconnect_distance = same_camera_reconnect_distance
        self.same_camera_reconnect_min_iou = same_camera_reconnect_min_iou
        self.reconnect_confirm_frames = reconnect_confirm_frames
        self.reconnect_confirm_ratio = reconnect_confirm_ratio
        self.reconnect_confirm_threshold = reconnect_confirm_threshold
        self.confirm_windows = confirm_windows
        self.revalidate_threshold = revalidate_threshold
        self.revalidate_windows = revalidate_windows
        self.pending_timeout = pending_timeout
        self.local_tracks = {}
        self.global_tracks = {}
        self.next_global_id = 1
        self._frame_camera = None
        self._frame_local_ids = set()

    def update_camera(self, camera_id, tracks, now):
        """Associate a complete camera frame with exclusive same-camera GIDs.

        Register every visible local ID before updating any of them. This avoids
        relying on wall-clock timeouts or detector iteration order for occupancy.
        Overlapping cameras may observe the same global identity simultaneously.
        """
        local_ids = {int(track['local_id']) for track in tracks}
        if len(local_ids) != len(tracks):
            raise ValueError('Duplicate local track IDs in a camera frame.')
        self._frame_camera = int(camera_id)
        self._frame_local_ids = local_ids
        result = {}
        def priority(track):
            state = self.local_tracks.get((int(camera_id), int(track['local_id'])))
            confirmed = state is not None and state.global_id is not None
            return (not confirmed, -track['feature_quality'], int(track['local_id']))
        try:
            for state in self.local_tracks.values():
                if state.camera_id == int(camera_id):
                    self._expire_candidate(state, now)
            # A previously lost local track can revive after its old GID was
            # reassigned. Resolve such existing conflicts before updating galleries.
            owners = {}
            for track in tracks:
                state = self.local_tracks.get((int(camera_id), int(track['local_id'])))
                if state is not None and state.global_id is not None:
                    owners.setdefault(state.global_id, []).append((state, track))
            for gid, contenders in owners.items():
                if len(contenders) < 2:
                    continue
                def owner_priority(item):
                    state, track = item
                    distance = self._distance(track['feature'], self.global_tracks[gid].features)
                    if track['feature_quality'] <= 0 or distance is None:
                        distance = float('inf')
                    return (distance, -track['feature_quality'], state.first_seen, state.local_id)
                contenders.sort(key=owner_priority)
                for state, _ in contenders[1:]:
                    self._reset_identity(state, gid, now)
                    self._log(state, 'same_camera_conflict', gid)
            for track in sorted(tracks, key=priority):
                result[track['local_id']] = self.update_track(
                    camera_id, track['local_id'], track['feature'], track['bbox'],
                    track['feature_quality'], now)
        finally:
            self._frame_camera = None
            self._frame_local_ids = set()
        return result

    def update_track(self, camera_id, local_id, feature, bbox, quality, now):
        key = (int(camera_id), int(local_id))
        state = self.local_tracks.get(key)
        if state is None:
            state = LocalTrackState(camera_id, local_id, now,
                                    self.track_feature_size, self.aggregate_frames)
            self.local_tracks[key] = state
        self._expire_candidate(state, now)
        accepted = state.add_observation(feature, bbox, quality, now)

        if state.global_id is not None:
            if accepted is not None:
                aggregated, weight = state.pop_aggregated_feature(self.aggregate_frames)
                if aggregated is not None:
                    self._revalidate(state, aggregated, weight, now)
            # Suspect frames cannot move the identity's last reliable position.
            if state.global_id is not None and not state.mismatch_windows:
                self._touch_global_position(state.global_id, state, now)
            return state.global_id if not state.mismatch_windows else None

        if accepted is None or not self._ready_to_confirm(state, now):
            return None
        if state.assignment_started is None:
            state.assignment_started = now
        if state.candidate_source == 'position':
            self._confirm_position(state, accepted, quality, now)
            return state.global_id

        if state.candidate_id is not None:
            distance = self._distance(accepted, self.global_tracks[state.candidate_id].features)
            if distance is not None and distance < self.threshold:
                state.candidate_progress = now

        aggregated, weight = state.pop_aggregated_feature(self.aggregate_frames)
        if aggregated is None:
            return None
        distances = self._candidates(state, aggregated, now)
        if distances and distances[0][1] < self.threshold:
            best_id, best_distance = distances[0]
            if len(distances) > 1 and distances[1][1] - best_distance < self.margin:
                state.clear_candidate()
                if now - state.assignment_started >= self.pending_timeout:
                    self._bind(state, self._new_global_id(state, now), now, 'new_after_timeout')
                    return state.global_id
                return None
            if state.candidate_id == best_id and state.candidate_source == 'reid':
                state.candidate_votes += 1
            else:
                state.clear_candidate()
                state.candidate_id = best_id
                state.candidate_source = 'reid'
                state.candidate_started = now
                state.candidate_votes = 1
            state.candidate_progress = now
            if state.candidate_votes >= self.confirm_windows:
                # Only the confirmation window is admitted, never earlier uncertain crops.
                self._add_global_feature(best_id, aggregated, weight)
                self._bind(state, best_id, now, 'reid_confirmed')
            return state.global_id

        state.clear_candidate()
        reconnect_id = self._same_camera_reconnect_id(state, now)
        if reconnect_id is not None:
            state.candidate_id = reconnect_id
            state.candidate_source = 'position'
            state.candidate_started = now
            state.candidate_progress = now
            return None
        self._bind(state, self._new_global_id(state, now), now, 'new')
        return state.global_id

    def _ready_to_confirm(self, state, now):
        return now - state.first_seen >= self.delay_seconds and len(state.features) >= self.min_features

    def _expire_candidate(self, state, now):
        if (state.candidate_id is not None and state.candidate_progress is not None
                and now - state.candidate_progress > self.pending_timeout):
            state.clear_candidate()
            state.clear_pending_features()
            state.features.clear()
            state.feature_qualities.clear()
            state.first_seen = now
            state.assignment_started = now

    def _candidates(self, state, feature, now):
        distances = []
        for global_id, identity in self.global_tracks.items():
            if global_id in state.blocked_ids or self._global_active_in_same_camera(global_id, state, now):
                continue
            distance = self._distance(feature, identity.features)
            if distance is not None:
                distances.append((global_id, distance))
        return sorted(distances, key=lambda item: (item[1], item[0]))

    def _revalidate(self, state, feature, quality, now):
        distance = self._distance(feature, self.global_tracks[state.global_id].features)
        if distance is None or distance > self.revalidate_threshold:
            state.mismatch_windows += 1
            state.suspect_features.append(feature)
            state.suspect_qualities.append(quality)
            if state.mismatch_windows >= self.revalidate_windows:
                old_id = state.global_id
                self._reset_identity(state, old_id, now)
                self._log(state, 'revalidate_rejected', old_id)
            return
        state.mismatch_windows = 0
        state.suspect_features = []
        state.suspect_qualities = []
        if now - state.last_global_update >= self.feature_update_interval:
            self._add_global_feature(state.global_id, feature, quality)
            state.last_global_update = now

    def _reset_identity(self, state, blocked_id, now):
        state.blocked_ids.add(blocked_id)
        state.global_id = None
        state.features.clear()
        state.feature_qualities.clear()
        # Recollect raw observations for reassignment; do not mix identities.
        state.clear_pending_features()
        state.clear_candidate()
        state.assignment_started = None
        state.first_seen = now
        state.mismatch_windows = 0
        state.suspect_features = []
        state.suspect_qualities = []

    def _confirm_position(self, state, feature, quality, now):
        candidate = state.candidate_id
        identity = self.global_tracks.get(candidate)
        if identity is None or self._global_active_in_same_camera(candidate, state, now):
            self._reject_position(state, now)
            return
        distance = self._distance(feature, identity.features)
        # Require separation from other eligible identities, too.
        alternatives = [distance for gid, distance in self._candidates(state, feature, now)
                        if gid != candidate]
        matched = (distance is not None and distance <= self.reconnect_confirm_threshold
                   and (not alternatives or min(alternatives) - distance >= self.margin))
        if matched:
            state.candidate_progress = now
        state.reconnect_features.append(feature)
        state.reconnect_qualities.append(quality)
        state.reconnect_matches.append(matched)
        if len(state.reconnect_matches) < self.reconnect_confirm_frames:
            return
        support = sum(state.reconnect_matches) / len(state.reconnect_matches)
        if support < self.reconnect_confirm_ratio:
            self._reject_position(state, now)
            return
        # Rejected observations must not contaminate the recovered gallery.
        good = [(f, q) for f, q, match in zip(state.reconnect_features,
                state.reconnect_qualities, state.reconnect_matches) if match]
        feature = mean_feature([f for f, _ in good], [q for _, q in good])
        self._add_global_feature(candidate, feature, float(np.mean([q for _, q in good])))
        self._bind(state, candidate, now, 'position_confirmed')

    def _reject_position(self, state, now):
        state.blocked_ids.add(state.candidate_id)
        rejected = state.candidate_id
        state.clear_candidate()
        state.clear_pending_features()
        state.features.clear()
        state.feature_qualities.clear()
        state.first_seen = now
        state.assignment_started = now
        # Resume appearance confirmation: a different existing GID may be correct.
        self._log(state, 'position_rejected', rejected)

    def _same_camera_reconnect_id(self, state, now):
        if not self.same_camera_reconnect or state.last_bbox is None:
            return None
        candidates = []
        for gid, identity in self.global_tracks.items():
            if gid in state.blocked_ids or self._global_active_in_same_camera(gid, state, now):
                continue
            previous = identity.camera_states.get(state.camera_id)
            if previous is None:
                continue
            age = now - previous['time']
            if not 0 <= age <= self.same_camera_reconnect_timeout:
                continue
            distance = bbox_center_distance_ratio(state.last_bbox, previous['bbox'])
            overlap = bbox_iou(state.last_bbox, previous['bbox'])
            if (distance <= self.same_camera_reconnect_distance or
                    (overlap > 0 and overlap >= self.same_camera_reconnect_min_iou)):
                candidates.append((distance - 0.5 * overlap + 0.01 * age, gid))
        return min(candidates)[1] if candidates else None

    def _global_active_in_same_camera(self, global_id, state, now):
        for other in self.local_tracks.values():
            if other is state or other.camera_id != state.camera_id:
                continue
            self._expire_candidate(other, now)
            if global_id not in (other.global_id, other.candidate_id):
                continue
            if self._frame_camera == state.camera_id:
                if other.local_id in self._frame_local_ids:
                    return True
            elif abs(now - other.last_seen) <= self.active_timeout:
                return True
        return False

    def _distance_values(self, query, gallery):
        fn = euclidean_distance if self.metric == 'euclidean' else cosine_distance
        return fn(query, gallery)

    def _distance(self, query, gallery):
        distances = self._distance_values(query, gallery)
        if distances is None or not len(distances):
            return None
        return float(np.mean(np.sort(distances)[:self.topk]))

    def _new_global_id(self, state, now):
        gid = self.next_global_id
        self.next_global_id += 1
        self.global_tracks[gid] = GlobalIdentity(gid, self.gallery_size, self.anchor_size, now)
        feature = mean_feature(state.features, state.feature_qualities)
        if feature is not None:
            self._add_global_feature(gid, feature, state.mean_quality(), force_anchor=True)
        return gid

    def _bind(self, state, global_id, now, action):
        state.global_id = global_id
        state.last_global_update = now
        state.clear_candidate()
        state.clear_pending_features()
        state.assignment_started = None
        self._touch_global_position(global_id, state, now)
        self._log(state, action, global_id)

    def _add_global_feature(self, global_id, feature, quality, force_anchor=False):
        if feature is None:
            return False
        feature = normalize_feature(feature)
        if feature is None:
            return False
        identity = self.global_tracks[global_id]
        if identity.features and not force_anchor:
            distances = self._distance_values(feature, identity.features)
            nearest = float(np.min(distances))
            if nearest > self.feature_update_max_distance or nearest < self.feature_min_novelty:
                return False
        if (force_anchor or quality >= self.anchor_min_confidence) and len(identity.anchor_features) < identity.anchor_size:
            identity.anchor_features.append(feature.copy())
        else:
            identity.recent_features.append(feature.copy())
        return True

    def _touch_global_position(self, global_id, state, now):
        if state.last_bbox is None:
            return
        identity = self.global_tracks[global_id]
        identity.last_seen = max(identity.last_seen, now)
        previous = identity.camera_states.get(state.camera_id)
        if previous is None or now >= previous['time']:
            identity.camera_states[state.camera_id] = {'bbox': state.last_bbox, 'time': now}

    def cleanup(self, now):
        for key, state in list(self.local_tracks.items()):
            if now - state.last_seen > self.stale_timeout:
                del self.local_tracks[key]

    @staticmethod
    def _log(state, action, global_id):
        print('[GlobalReID] camera={} local={} action={} assigned_gid={}'.format(
            state.camera_id, state.local_id, action, global_id))
