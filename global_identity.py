"""Online identity association, independent of video capture and neural models.

Only confirmed, appearance-consistent observations may update a global gallery.
Use update_camera() to reserve identities for all people in a camera frame.
"""

from collections import deque
import math

import numpy as np

from offline_association import RepresentativeGallery, feature_distance


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
        self.candidate_features = deque(maxlen=max_features)
        self.candidate_qualities = deque(maxlen=max_features)
        self.candidate_matches = deque(maxlen=max_features)
        self.candidate_rank_ids = deque(maxlen=max_features)
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
        self.last_candidates = []

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
        self.candidate_features.clear()
        self.candidate_qualities.clear()
        self.candidate_matches.clear()
        self.candidate_rank_ids.clear()
        self.reconnect_features = []
        self.reconnect_qualities = []
        self.reconnect_matches = []


class GlobalIdentity:
    def __init__(self, global_id, gallery_size, anchor_size, now,
                 prototype_count=6, prototype_merge_threshold=.15):
        self.global_id = int(global_id)
        self.anchor_size = anchor_size
        self.anchor_features = []
        self.anchor_qualities = []
        self.recent_features = deque(maxlen=gallery_size - anchor_size)
        self.recent_qualities = deque(maxlen=gallery_size - anchor_size)
        self.max_prototypes = max(0, int(prototype_count))
        self.prototype_merge_threshold = float(prototype_merge_threshold)
        self.prototype_features = []
        self.prototype_masses = []
        self.prototype_qualities = []
        self.prototype_counts = []
        self.created_at = self.last_seen = float(now)
        self.camera_states = {}

    @property
    def features(self):
        return self.anchor_features + list(self.recent_features)

    @property
    def feature_qualities(self):
        return self.anchor_qualities + list(self.recent_qualities)

    @property
    def match_features(self):
        return self.prototype_features or self.features

    @property
    def match_qualities(self):
        return self.prototype_qualities or self.feature_qualities

    def update_prototypes(self, feature, quality):
        feature = normalize_feature(feature)
        if self.max_prototypes <= 0 or feature is None:
            return False
        quality = max(float(quality), 1e-6)
        if not self.prototype_features:
            self.prototype_features.append(feature.copy())
            self.prototype_masses.append(quality)
            self.prototype_qualities.append(quality)
            self.prototype_counts.append(1)
            return True
        distances = cosine_distance(feature, self.prototype_features)
        nearest = int(np.argmin(distances))
        if float(distances[nearest]) <= self.prototype_merge_threshold:
            old_mass = self.prototype_masses[nearest]
            new_mass = old_mass + quality
            merged = (self.prototype_features[nearest] * old_mass + feature * quality) / new_mass
            self.prototype_features[nearest] = normalize_feature(merged)
            count = self.prototype_counts[nearest]
            self.prototype_qualities[nearest] = (
                self.prototype_qualities[nearest] * count + quality) / (count + 1)
            self.prototype_counts[nearest] = count + 1
            self.prototype_masses[nearest] = new_mass
            return True
        if len(self.prototype_features) < self.max_prototypes:
            self.prototype_features.append(feature.copy())
            self.prototype_masses.append(quality)
            self.prototype_qualities.append(quality)
            self.prototype_counts.append(1)
            return True
        return False


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
        revalidate_reassign_threshold=.20, revalidate_reassign_margin=.15,
        revalidate_adapt_threshold=.55,
        pending_timeout=10.0, match_strategy='topk', candidate_threshold=None,
        strong_threshold=None, borderline_confirm_frames=15,
        borderline_confirm_ratio=.8, borderline_confirm_threshold=None,
        prototype_count=6, prototype_merge_threshold=.15,
        same_camera_reconnect_reid_threshold=.5,
        same_camera_reconnect_reid_margin=.05,
        same_camera_conflict_continuity=5.0, decision_log=None
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
               feature_min_novelty, same_camera_reconnect_distance,
               same_camera_conflict_continuity) < 0:
            raise ValueError('Timeouts, margins and distances must be nonnegative.')
        if not 0 < reconnect_confirm_ratio <= 1:
            raise ValueError('reconnect_confirm_ratio must be in (0, 1].')
        if not 0 <= same_camera_reconnect_min_iou <= 1:
            raise ValueError('same_camera_reconnect_min_iou must be in [0, 1].')
        if not 0 <= anchor_min_confidence <= 1:
            raise ValueError('anchor_min_confidence must be in [0, 1].')
        if feature_min_novelty > feature_update_max_distance:
            raise ValueError('Feature novelty threshold exceeds update distance.')
        if match_strategy not in ('topk', 'centroid', 'hybrid', 'adaptive', 'bidirectional'):
            raise ValueError('Unknown online gallery matching strategy.')
        if candidate_threshold is None:
            candidate_threshold = threshold
        if strong_threshold is not None and (
                not np.isfinite(strong_threshold) or strong_threshold <= 0
                or strong_threshold > candidate_threshold):
            raise ValueError('Strong threshold must be positive and <= candidate threshold.')
        if borderline_confirm_threshold is None:
            borderline_confirm_threshold = threshold
        if (not np.isfinite(candidate_threshold) or candidate_threshold <= 0
                or not 0 < borderline_confirm_threshold <= candidate_threshold):
            raise ValueError('Candidate and borderline thresholds are inconsistent.')
        if borderline_confirm_frames < 1 or not 0 < borderline_confirm_ratio <= 1:
            raise ValueError('Borderline confirmation settings are invalid.')
        if prototype_count < 0 or not 0 <= prototype_merge_threshold <= 2:
            raise ValueError('Prototype settings are invalid.')
        self.match_strategy = match_strategy
        self.candidate_threshold = candidate_threshold
        self.strong_threshold = strong_threshold
        self.borderline_confirm_frames = int(borderline_confirm_frames)
        self.borderline_confirm_ratio = float(borderline_confirm_ratio)
        self.borderline_confirm_threshold = float(borderline_confirm_threshold)
        self.prototype_count = int(prototype_count)
        self.prototype_merge_threshold = float(prototype_merge_threshold)
        self.same_camera_reconnect_reid_threshold = float(same_camera_reconnect_reid_threshold)
        self.same_camera_reconnect_reid_margin = float(same_camera_reconnect_reid_margin)
        self.same_camera_conflict_continuity = float(same_camera_conflict_continuity)
        self.decision_log = decision_log
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
        self.revalidate_reassign_threshold = float(revalidate_reassign_threshold)
        self.revalidate_reassign_margin = float(revalidate_reassign_margin)
        self.revalidate_adapt_threshold = float(revalidate_adapt_threshold)
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
                    # Prefer the continuously visible owner over a stale local
                    # track that has just revived after a long detector gap.
                    gap = max(0.0, now - state.last_seen)
                    revived = gap > self.same_camera_conflict_continuity
                    distance = self._identity_distance(track['feature'], self.global_tracks[gid])
                    if track['feature_quality'] <= 0 or distance is None:
                        distance = float('inf')
                    return (revived, distance, gap, -track['feature_quality'],
                            state.first_seen, state.local_id)
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
                samples = list(state.pending_features)
                qualities = list(state.pending_qualities)
                aggregated, weight = state.pop_aggregated_feature(self.aggregate_frames)
                if aggregated is not None:
                    self._revalidate(state, aggregated, weight, now, samples, qualities)
            # Suspect frames cannot move the identity's last reliable position.
            if state.global_id is not None and not state.mismatch_windows:
                self._touch_global_position(state.global_id, state, now)
            # Keep an established identity visible while its gallery is frozen.
            # A moderate pose change must not make the GID flicker or disappear.
            return state.global_id

        if accepted is None or not self._ready_to_confirm(state, now):
            return None
        if state.assignment_started is None:
            state.assignment_started = now
        # Same-camera continuity is stronger evidence than a generic gallery
        # candidate.  Evaluate it first so a previously fragmented duplicate
        # identity cannot hide the actual predecessor.
        if state.candidate_id is None:
            reconnect_id = self._same_camera_reconnect_id(state, now)
            if reconnect_id is not None:
                state.candidate_id = reconnect_id
                state.candidate_source = 'position'
                state.candidate_started = now
                state.candidate_progress = now
                return None
        if state.candidate_source == 'position':
            self._confirm_position(state, accepted, quality, now)
            return state.global_id

        window = list(state.pending_features)
        window_weights = list(state.pending_qualities)
        aggregated, weight = state.pop_aggregated_feature(self.aggregate_frames)
        if aggregated is None:
            return None
        distances = self._candidates(state, aggregated, now, window, window_weights)
        best = distances[0] if distances else None

        # With an explicit strong threshold, a clear high-confidence match can
        # bind immediately. Wider or ambiguous candidates remain private until
        # enough later views support the same identity.
        started_now = False
        if state.candidate_id is None and best and best[1] < self.candidate_threshold:
            separated = len(distances) == 1 or distances[1][1] - best[1] >= self.margin
            if (self.strong_threshold is not None and best[1] <= self.strong_threshold
                    and separated):
                self._add_global_feature(best[0], aggregated, weight)
                state.last_candidates = distances
                self._bind(state, best[0], now, 'reid_strong')
                return state.global_id
            state.candidate_id = best[0]
            state.candidate_source = 'reid'
            state.candidate_started = state.candidate_progress = now
            started_now = self.strong_threshold is not None

        if state.candidate_id is not None and state.candidate_source == 'reid':
            candidate = state.candidate_id
            if started_now:
                self._log(state, 'borderline_pending', None, candidates=distances,
                          confirm_samples=0)
                return None
            state.candidate_votes += 1
            state.candidate_features.extend(window)
            state.candidate_qualities.extend(window_weights)
            # Count frame-level rank-1 support instead of trusting one averaged
            # crop. This is the stable part of the original temporal gate.
            for sample in window:
                ranked = self._candidates(state, sample, now)
                rank_id = ranked[0][0] if ranked else None
                state.candidate_rank_ids.append(rank_id)
                state.candidate_matches.append(rank_id == candidate)
            # Active valid evidence keeps a pending decision alive even when a
            # changing viewpoint temporarily changes the rank-1 candidate.
            state.candidate_progress = now

            required = (self.confirm_windows if self.strong_threshold is None
                        else self.borderline_confirm_frames)
            observed = (state.candidate_votes if self.strong_threshold is None
                        else len(state.candidate_matches))
            if observed >= required:
                evidence = mean_feature(state.candidate_features, state.candidate_qualities)
                final = self._candidates(state, evidence, now,
                                         state.candidate_features, state.candidate_qualities)
                final_candidate = final[0][0] if final else candidate
                support = (sum(gid == final_candidate for gid in state.candidate_rank_ids)
                           / len(state.candidate_rank_ids)
                           if state.candidate_rank_ids else 0.0)
                final_limit = (self.threshold if self.strong_threshold is None
                               else self.borderline_confirm_threshold)
                separated = (final and
                    (len(final) == 1 or final[1][1] - final[0][1] >= self.margin))
                strong_alternative = (self.strong_threshold is not None and final
                    and final[0][0] != candidate and final[0][1] <= self.strong_threshold
                    and (len(final) == 1 or final[1][1] - final[0][1] >= self.margin))
                ratio_ok = (True if self.strong_threshold is None
                            else support >= self.borderline_confirm_ratio)
                # Stable frame-level support can resolve nearly identical old
                # galleries without minting another duplicate GID.
                stable_winner = (self.strong_threshold is not None and final
                    and support >= max(.9, self.borderline_confirm_ratio))
                if strong_alternative:
                    replacement = final[0][0]
                    self._admit_candidate_features(replacement, state, evidence)
                    self._bind(state, replacement, now, 'borderline_reassigned_strong')
                elif (separated or stable_winner) and ratio_ok and final[0][1] <= final_limit:
                    self._admit_candidate_features(final_candidate, state, evidence)
                    action = ('borderline_confirmed' if final_candidate == candidate
                              else 'borderline_reassigned')
                    self._bind(state, final_candidate, now, action)
                else:
                    max_observed = state.candidate_matches.maxlen
                    if (self.strong_threshold is not None and final
                            and final[0][1] <= self.candidate_threshold
                            and observed < max_observed):
                        # Let close candidates separate over a longer window;
                        # short-lived ambiguity must not create another gallery.
                        state.candidate_id = final[0][0]
                        state.candidate_progress = now
                        self._log(state, 'borderline_extended', None,
                                  candidates=final, support_ratio=support,
                                  confirm_samples=observed)
                        return None
                    details = dict(candidates=final, support_ratio=support,
                                   confirm_samples=observed)
                    self._log(state, 'borderline_rejected', None, **details)
                    state.clear_candidate()
                    self._bind(state, self._new_global_id(state, now), now,
                               'new_after_borderline')
            elif (self.strong_threshold is None and
                    now - state.assignment_started >= self.pending_timeout):
                self._bind(state, self._new_global_id(state, now), now, 'new_after_timeout')
            else:
                self._log(state, 'borderline_pending', None, candidates=distances,
                          confirm_samples=observed)
            return state.global_id

        state.clear_candidate()
        self._bind(state, self._new_global_id(state, now), now, 'new')
        return state.global_id

    def _admit_candidate_features(self, global_id, state, evidence):
        if self.match_strategy == 'bidirectional':
            for feature, quality in zip(state.candidate_features, state.candidate_qualities):
                self._add_global_feature(global_id, feature, quality)
        else:
            self._add_global_feature(global_id, evidence,
                                     float(np.mean(state.candidate_qualities)))

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
            # Discard stale evidence without extending the assignment deadline.
            # Otherwise repeated short gaps can keep an ambiguous person pending forever.

    def _candidates(self, state, feature, now, features=None, qualities=None):
        distances = []
        for global_id, identity in self.global_tracks.items():
            if global_id in state.blocked_ids or self._global_active_in_same_camera(global_id, state, now):
                continue
            if self.match_strategy == 'bidirectional' and features is not None and len(features) >= 2:
                query = RepresentativeGallery(list(features), list(qualities) if qualities is not None else None,
                                               limit=12, metric=self.metric)
                gallery = RepresentativeGallery(identity.features, limit=12, metric=self.metric)
                distance = query.distance(gallery, 'bidirectional')
            else:
                distance = self._identity_distance(feature, identity)
            if distance is not None and np.isfinite(distance):
                distances.append((global_id, distance))
        state.last_candidates = sorted(distances, key=lambda item: (item[1], item[0]))
        return state.last_candidates

    def _revalidate(self, state, feature, quality, now, samples=None, qualities=None):
        distance = self._identity_distance(feature, self.global_tracks[state.global_id])
        if self.match_strategy == 'bidirectional' and samples:
            # A confirmed person may show only ONE known view now. Requiring
            # this short window to cover every historical view would cause
            # false rejection on turns. Query-to-gallery support is sufficient
            # for retention; initial cross-camera binding remains bidirectional.
            distances = feature_distance(samples, self.global_tracks[state.global_id].features, self.metric)
            distance = float(np.average(distances.min(axis=1), weights=qualities))
        if distance is None or distance > self.revalidate_threshold:
            state.mismatch_windows += 1
            self._log(state, 'gallery_frozen', state.global_id, distance=distance,
                      mismatch_windows=state.mismatch_windows)
            state.suspect_features.append(feature)
            state.suspect_qualities.append(quality)
            if state.mismatch_windows >= self.revalidate_windows:
                current_id = state.global_id
                suspect = mean_feature(state.suspect_features, state.suspect_qualities)
                ranked = [(gid, value) for gid, value in self._candidates(state, suspect, now)
                          if gid != current_id]
                alternative = ranked[0] if ranked else None
                separated = (alternative is not None and
                    (len(ranked) == 1 or ranked[1][1] - alternative[1] >= self.margin))
                should_reassign = (separated
                    and alternative[1] <= self.revalidate_reassign_threshold
                    and distance - alternative[1] >= self.revalidate_reassign_margin)
                if should_reassign:
                    state.blocked_ids.add(current_id)
                    state.global_id = alternative[0]
                    state.last_global_update = now
                    self._add_global_feature(alternative[0], suspect,
                                             float(np.mean(state.suspect_qualities)))
                    self._touch_global_position(alternative[0], state, now)
                    self._log(state, 'revalidate_reassigned', alternative[0],
                              previous_global_id=current_id, distance=distance,
                              alternative_distance=alternative[1])
                else:
                    # Continuous LID evidence is stronger than a moderate pose
                    # distance. Keep the visible identity stable and learn the
                    # new view only inside a conservative adaptation ceiling.
                    if distance is not None and distance <= self.revalidate_adapt_threshold:
                        self._add_global_feature(current_id, suspect,
                            float(np.mean(state.suspect_qualities)), allow_novel=True)
                    self._log(state, 'revalidate_retained', current_id,
                              distance=distance, candidates=ranked[:3])
                state.mismatch_windows = 0
                state.suspect_features = []
                state.suspect_qualities = []
            return
        state.mismatch_windows = 0
        state.suspect_features = []
        state.suspect_qualities = []
        if now - state.last_global_update >= self.feature_update_interval:
            if self.match_strategy == 'bidirectional' and samples:
                for sample, weight in zip(samples, qualities):
                    self._add_global_feature(state.global_id, sample, weight,
                                             allow_novel=True)
            else:
                self._add_global_feature(state.global_id, feature, quality,
                                         allow_novel=True)
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
        distance = self._identity_distance(feature, identity)
        # Require separation from other eligible identities, too.
        alternatives = [distance for gid, distance in self._candidates(state, feature, now)
                        if gid != candidate]
        # Spatial continuity may break a near tie against a duplicate gallery.
        # A substantially better appearance alternative still rejects it.
        matched = (distance is not None and distance <= self.reconnect_confirm_threshold
                   and (not alternatives or min(alternatives)
                        + self.same_camera_reconnect_reid_margin >= distance))
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
        query = mean_feature(state.features, state.feature_qualities)
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
                appearance = self._identity_distance(query, identity)
                if appearance is not None and appearance <= self.same_camera_reconnect_reid_threshold:
                    candidates.append((appearance, distance - .5*overlap + .01*age, gid))
        candidates.sort()
        if not candidates or (len(candidates) > 1 and
                candidates[1][0]-candidates[0][0] < self.same_camera_reconnect_reid_margin):
            return None
        return candidates[0][2]

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

    def _identity_distance(self, query, identity, query_features=None, qualities=None):
        if (self.match_strategy == 'bidirectional' and query_features is not None
                and len(query_features) >= 2 and identity.features):
            query_gallery = RepresentativeGallery(
                list(query_features), list(qualities) if qualities is not None else None,
                                                   limit=12, metric=self.metric)
            identity_gallery = RepresentativeGallery(identity.features,
                                                      identity.feature_qualities,
                                                      limit=12, metric=self.metric)
            return query_gallery.distance(identity_gallery, 'bidirectional')
        mode = 'adaptive' if self.match_strategy == 'bidirectional' else self.match_strategy
        return self._distance(query, identity.match_features, identity.match_qualities,
                              topk=1 if identity.prototype_features else None,
                              match_mode=mode)

    def _distance(self, query, gallery, qualities=None, topk=None, match_mode=None):
        distances = self._distance_values(query, gallery)
        if distances is None or not len(distances):
            return None
        mode = self.match_strategy if match_mode is None else match_mode
        if mode == 'bidirectional':
            mode = 'adaptive'
        effective_topk = self.topk if topk is None else max(1, int(topk))
        topk_distance = float(np.mean(np.sort(distances)[:effective_topk]))
        if mode == 'topk':
            return topk_distance
        centroid = mean_feature(gallery, qualities)
        if centroid is None:
            return None
        if self.metric == 'euclidean':
            centroid_distance = float(np.linalg.norm(normalize_feature(query) - centroid))
        else:
            centroid_distance = float(1.0 - np.dot(normalize_feature(query), centroid))
        if mode == 'centroid':
            return centroid_distance
        if mode == 'adaptive' and self.strong_threshold is not None and topk_distance <= self.strong_threshold:
            return topk_distance
        return max(topk_distance, centroid_distance)

    def _new_global_id(self, state, now):
        gid = self.next_global_id
        self.next_global_id += 1
        self.global_tracks[gid] = GlobalIdentity(
            gid, self.gallery_size, self.anchor_size, now,
            self.prototype_count, self.prototype_merge_threshold)
        feature = mean_feature(state.features, state.feature_qualities)
        if feature is not None:
            if self.match_strategy == 'bidirectional':
                for sample, quality in zip(state.features, state.feature_qualities):
                    self._add_global_feature(gid, sample, quality, force_anchor=True)
            else:
                self._add_global_feature(gid, feature, state.mean_quality(), force_anchor=True)
        return gid

    def _bind(self, state, global_id, now, action):
        details = dict(candidates=state.last_candidates[:3], confirm_windows=state.candidate_votes,
                       candidate_id=state.candidate_id)
        state.global_id = global_id
        state.last_global_update = now
        state.clear_candidate()
        state.clear_pending_features()
        state.assignment_started = None
        self._touch_global_position(global_id, state, now)
        self._log(state, action, global_id, **details)

    def _add_global_feature(self, global_id, feature, quality, force_anchor=False,
                            allow_novel=False):
        if feature is None:
            return False
        feature = normalize_feature(feature)
        if feature is None:
            return False
        identity = self.global_tracks[global_id]
        prototype_updated = False
        if identity.features:
            update_distance = self._identity_distance(feature, identity)
            distances = self._distance_values(feature, identity.features)
            nearest = float(np.min(distances))
            if (not force_anchor and not allow_novel and
                    (update_distance is None or update_distance > self.feature_update_max_distance)):
                return False
            identity.update_prototypes(feature, quality)
            prototype_updated = True
            if nearest < self.feature_min_novelty:
                return True
        if (force_anchor or quality >= self.anchor_min_confidence) and len(identity.anchor_features) < identity.anchor_size:
            identity.anchor_features.append(feature.copy())
            identity.anchor_qualities.append(float(quality))
        else:
            identity.recent_features.append(feature.copy())
            identity.recent_qualities.append(float(quality))
        if not prototype_updated:
            identity.update_prototypes(feature, quality)
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

    def _log(self, state, action, global_id, **details):
        if self.decision_log is not None:
            self.decision_log(dict(camera_id=state.camera_id, local_id=state.local_id,
                timestamp=state.last_seen, action=action, global_id=global_id,
                strategy=self.match_strategy, threshold=self.threshold,
                candidate_threshold=self.candidate_threshold, margin=self.margin, **details))
        if action not in ('reid_pending', 'borderline_pending', 'borderline_extended'):
            print('[GlobalReID] camera={} local={} action={} assigned_gid={}'.format(
                state.camera_id, state.local_id, action, global_id))
