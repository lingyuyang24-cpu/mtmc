"""Identity-level regressions, independent of clothing, camera layout and models."""
import contextlib
import io
import math
import unittest

import numpy as np

from deep_sort.detection import Detection
from deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.tracker import Tracker
from global_identity import GlobalIDManager
from offline_association import RepresentativeGallery

A, B = np.eye(2, dtype=np.float32)
BOX = (20, 20, 60, 100)


class MatchingStabilityTests(unittest.TestCase):
    def setUp(self):
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)

    def manager(self, **options):
        config = dict(delay_seconds=0, min_features=2, aggregate_frames=2,
                      confirm_windows=2, revalidate_windows=2,
                      same_camera_reconnect=False, feature_update_interval=0)
        config.update(options)
        return GlobalIDManager(**config)

    def feed(self, manager, features, camera=0, start=0, local=1):
        return [manager.update_track(camera, local, feature, BOX, .9, start+i*.1)
                for i, feature in enumerate(features)]

    def test_low_score_can_continue_but_cannot_seed_track(self):
        tracker = Tracker(NearestNeighborDistanceMetric('cosine', .2, 100), n_init=2,
                          new_track_min_confidence=.25)
        for score in (.9, .9, .15, .15):
            detections = [Detection([20, 20, 40, 80], score, A),
                          Detection([100, 20, 30, 80], .12, B)]
            tracker.predict()
            tracker.update(detections)
        self.assertEqual([t.track_id for t in tracker.tracks], [1])
        self.assertEqual(tracker.tracks[0].hits, 4)

    def test_multiview_match_needs_support_in_both_directions(self):
        diverse = RepresentativeGallery([A, B])
        self.assertAlmostEqual(diverse.distance(diverse, 'bidirectional'), 0)
        self.assertAlmostEqual(diverse.distance(diverse, 'pairwise'), .5)
        unrelated = RepresentativeGallery([A, -A, -B])
        self.assertGreater(diverse.distance(unrelated, 'bidirectional'), .3)

    def test_hybrid_requires_nearest_view_and_identity_center(self):
        manager = self.manager(match_strategy='hybrid', prototype_count=0, topk=1)
        self.feed(manager, [A, A])
        manager._add_global_feature(1, B, .9, force_anchor=True)
        hybrid = manager._identity_distance(A, manager.global_tracks[1])
        manager.match_strategy = 'topk'
        topk = manager._identity_distance(A, manager.global_tracks[1])
        self.assertAlmostEqual(topk, 0.0)
        self.assertGreater(hybrid, .2)

    def test_explicit_strong_match_is_immediate(self):
        manager = self.manager(strong_threshold=.1, candidate_threshold=.4,
                               borderline_confirm_frames=6)
        self.feed(manager, [A, A])
        self.assertEqual(self.feed(manager, [A, A], camera=1, start=1), [None, 1])

    def test_borderline_match_waits_for_frame_support(self):
        manager = self.manager(strong_threshold=.1, candidate_threshold=.4,
                               borderline_confirm_threshold=.3,
                               borderline_confirm_frames=6,
                               borderline_confirm_ratio=.8)
        self.feed(manager, [A, A])
        weak = [0.72, math.sqrt(1-.72**2)]
        result = self.feed(manager, [weak] * 8, camera=1, start=1)
        self.assertNotIn(1, result[:7])
        self.assertEqual(result[-1], 1)

    def test_consistent_distance_034_reuses_gid_after_confirmation(self):
        manager = self.manager(strong_threshold=.25, candidate_threshold=.45,
                               borderline_confirm_threshold=.35,
                               borderline_confirm_frames=6,
                               borderline_confirm_ratio=.8)
        self.feed(manager, [A, A])
        changed_view = [.66, math.sqrt(1-.66**2)]  # cosine distance .34
        result = self.feed(manager, [changed_view] * 8, camera=1, start=1)
        self.assertNotIn(1, result[:7])
        self.assertEqual(result[-1], 1)
        self.assertEqual(len(manager.global_tracks), 1)

    def test_borderline_can_switch_to_later_clear_strong_identity(self):
        manager = self.manager(strong_threshold=.1, candidate_threshold=.4,
                               borderline_confirm_threshold=.3,
                               borderline_confirm_frames=4,
                               borderline_confirm_ratio=.8)
        self.feed(manager, [A, A], local=1)
        self.feed(manager, [B, B], local=2)
        weak_a = [.72, math.sqrt(1-.72**2)]
        self.feed(manager, [weak_a, weak_a], camera=1, start=1)
        result = self.feed(manager, [B, B, B, B], camera=1, start=2)
        self.assertEqual(result[-1], 2)

    def test_confirmed_identity_stays_stable_across_known_views(self):
        manager = self.manager(match_strategy='bidirectional', min_features=4)
        self.assertEqual(self.feed(manager, [A, B, A, B])[-1], 1)
        for start, features in ((1, [A]*8), (2, [B]*8), (3, [A, B]*4)):
            self.assertEqual(self.feed(manager, features, start=start), [1]*8)
        self.assertEqual(len(manager.global_tracks), 1)

    def test_continuous_pose_drift_is_learned_and_reconnects_after_gap(self):
        manager = self.manager(match_strategy='adaptive', strong_threshold=.25,
                               candidate_threshold=.45, revalidate_threshold=.4,
                               revalidate_windows=2, revalidate_adapt_threshold=.55)
        self.feed(manager, [A, A])
        close_view = [0.57, math.sqrt(1-.57**2)]  # cosine distance about .43
        self.assertEqual(self.feed(manager, [close_view] * 4, start=1), [1, 1, 1, 1])
        self.assertGreaterEqual(len(manager.global_tracks[1].features), 2)
        # A new LID showing only the close view should recover the old GID.
        self.assertEqual(self.feed(manager, [close_view, close_view], local=2, start=10),
                         [None, 1])

    def test_multiview_cross_camera_confirmation_preserves_separate_views(self):
        manager = self.manager(match_strategy='bidirectional', min_features=4)
        self.feed(manager, [A, B, A, B])
        result = self.feed(manager, [B, A]*4, camera=1, start=1)
        self.assertEqual(result[-1], 1)
        self.assertEqual(len(manager.global_tracks), 1)
        self.assertGreaterEqual(len(manager.global_tracks[1].features), 2)

    def test_weak_start_can_confirm_with_later_evidence_without_pollution(self):
        manager = self.manager(candidate_threshold=.4)
        self.feed(manager, [A, A])
        weak = [0.65, math.sqrt(1-.65**2)]  # distance .35: candidate only
        self.assertEqual(self.feed(manager, [weak, weak], camera=1, start=1), [None, None])
        np.testing.assert_array_equal(manager.global_tracks[1].features, [A])
        self.assertEqual(self.feed(manager, [A, A], camera=1, start=2)[-1], 1)

    def test_persistently_weak_candidate_does_not_gain_id_by_waiting(self):
        manager = self.manager(candidate_threshold=.4, pending_timeout=1)
        self.feed(manager, [A, A])
        weak = [0.65, math.sqrt(1-.65**2)]
        result = self.feed(manager, [weak]*20, camera=1, start=1)
        self.assertNotIn(1, result)
        self.assertEqual(result[-1], 2)
        np.testing.assert_array_equal(manager.global_tracks[1].features, [A])

    def test_reconnect_uses_appearance_before_nearest_position(self):
        manager = self.manager(threshold=.1, same_camera_reconnect=True)
        self.feed(manager, [A, A], local=1)
        # Second person in the same camera, away from the first person.
        for now in (0, .1):
            manager.update_track(0, 2, B, (100, 20, 140, 100), .9, now)
        manager.global_tracks[2].camera_states[0]['bbox'] = (25, 20, 65, 100)
        feature = [math.sqrt(1-.75**2), .75]
        self.feed(manager, [feature, feature], local=3, start=1)
        self.assertEqual(manager.local_tracks[(0, 3)].candidate_id, 2)


if __name__ == '__main__':
    unittest.main()
