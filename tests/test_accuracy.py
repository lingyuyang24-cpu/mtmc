"""Regression scenarios for detection, local tracking and online global IDs."""

import contextlib
import io
import math
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from deep_sort.preprocessing import (
    delete_overlap_box, non_max_suppression, select_detection_indices)
from global_identity import GlobalIDManager, mean_feature
from demo_stream import (CameraTracker, parse_args, reid_feature_quality,
                         suppress_duplicate_tracks)


A = np.array([1.0, 0.0], dtype=np.float32)
B = np.array([0.0, 1.0], dtype=np.float32)
BOX = (20, 20, 60, 100)


def observation(local_id, feature=A, quality=0.9, bbox=BOX):
    return dict(local_id=local_id, feature=feature, feature_quality=quality, bbox=bbox)


class DetectionTests(unittest.TestCase):
    def test_duplicate_boxes_keep_high_score(self):
        boxes = np.array([[10, 10, 40, 80], [10, 10, 40, 80]])
        self.assertEqual(delete_overlap_box(boxes, 0.4, [0.6, 0.95]), [1])

    def test_partial_overlap_uses_iou_not_coverage(self):
        boxes = np.array([[0, 0, 100, 100], [50, 0, 100, 100]])
        self.assertEqual(set(non_max_suppression(boxes, 0.4, [0.9, 0.8])), {0, 1})

    def test_default_preserves_detector_output(self):
        boxes = [[0, 0, 100, 100], [20, 0, 100, 100]]
        self.assertEqual(select_detection_indices(boxes, [0.9, 0.8]), [0, 1])
        self.assertEqual(select_detection_indices(boxes, [0.9, 0.8], 'nms'), [0])

    def test_empty_detections(self):
        self.assertEqual(delete_overlap_box([], 0.4), [])


class GlobalIdentityTests(unittest.TestCase):
    def setUp(self):
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)
        self.manager = self.make_manager()

    @staticmethod
    def make_manager(**kwargs):
        settings = dict(delay_seconds=0, min_features=2, aggregate_frames=2,
                        confirm_windows=2, revalidate_windows=2,
                        track_feature_size=10, same_camera_reconnect=False,
                        pending_timeout=100, feature_update_interval=0)
        settings.update(kwargs)
        return GlobalIDManager(**settings)

    def feed(self, camera=0, local=1, feature=A, start=0, count=2, manager=None):
        manager = manager or self.manager
        return [manager.update_track(camera, local, feature, BOX, 0.9, start + i * 0.1)
                for i in range(count)]

    def test_initial_identity_requires_enough_features(self):
        self.assertEqual(self.feed(), [None, 1])

    def test_cross_camera_match_waits_for_independent_windows(self):
        self.feed()
        self.assertEqual(self.feed(camera=1, start=1, count=4), [None, None, None, 1])

    def test_pending_match_cannot_pollute_gallery(self):
        self.feed()
        before = np.asarray(self.manager.global_tracks[1].features).copy()
        self.feed(camera=1, feature=[0.95, 0.2], start=1, count=2)
        np.testing.assert_array_equal(before, self.manager.global_tracks[1].features)

    def test_different_person_gets_new_identity(self):
        self.feed()
        self.assertEqual(self.feed(camera=1, feature=B, start=1), [None, 2])

    def test_same_camera_occupancy_is_frame_based_even_at_low_fps(self):
        self.feed()
        # The established person is deliberately last in detector order. Seconds
        # have elapsed since its last frame, exceeding the old 0.3s timeout.
        tracks = [observation(2), observation(1, None, 0)]
        self.manager.update_camera(0, tracks, 10)
        result = self.manager.update_camera(0, tracks, 11)
        self.assertEqual(result[1], 1)
        self.assertEqual(result[2], 2)

    def test_simultaneous_pending_people_cannot_share_gid(self):
        self.feed()
        tracks = [observation(2), observation(1)]
        for now in (1, 1.1, 1.2, 1.3):
            result = self.manager.update_camera(1, tracks, now)
        self.assertEqual(len(set(result.values())), 2)
        self.assertNotIn(None, result.values())

    def test_revived_track_cannot_share_reassigned_gid(self):
        self.feed()
        # Track 1 disappears, track 2 inherits its identity, then track 1 revives.
        self.assertEqual(self.feed(local=2, start=1, count=4)[-1], 1)
        result = self.manager.update_camera(0, [observation(2), observation(1)], 3)
        self.assertEqual(sum(gid == 1 for gid in result.values()), 1)
        confirmed = [gid for gid in result.values() if gid is not None]
        self.assertEqual(len(confirmed), len(set(confirmed)))

    def test_continuous_owner_keeps_gid_when_stale_local_track_revives(self):
        self.feed()
        self.assertEqual(self.feed(local=2, start=1, count=4)[-1], 1)
        result = self.manager.update_camera(0, [observation(1), observation(2)], 20)
        self.assertEqual(result[2], 1)
        self.assertNotEqual(result[1], 1)

    def test_revalidation_freezes_gallery_but_retains_incumbent_without_alternative(self):
        self.feed()
        before = np.asarray(self.manager.global_tracks[1].features).copy()
        results = self.feed(feature=B, start=1, count=4)
        self.assertEqual(results, [1, 1, 1, 1])
        self.assertEqual(self.manager.local_tracks[(0, 1)].global_id, 1)
        np.testing.assert_array_equal(before, self.manager.global_tracks[1].features)

    def test_revalidation_reassigns_only_to_clear_strong_alternative(self):
        self.feed()
        self.feed(camera=1, local=1, feature=B)
        results = self.feed(feature=B, start=1, count=4)
        self.assertEqual(results[-1], 2)

    def test_one_bad_window_can_recover(self):
        self.feed()
        self.assertEqual(self.feed(feature=B, start=1)[-1], 1)
        self.assertEqual(self.feed(feature=A, start=2)[-1], 1)

    def test_position_reconnect_not_visible_before_confirmation(self):
        manager = self.make_manager(threshold=0.1, same_camera_reconnect=True,
                                    reconnect_confirm_frames=4)
        self.feed(manager=manager)
        feature = [0.75, math.sqrt(1 - 0.75 ** 2)]
        self.assertEqual(self.feed(local=2, feature=feature, start=1, manager=manager),
                         [None, None])
        self.assertIsNone(manager.local_tracks[(0, 2)].global_id)
        results = self.feed(local=2, feature=feature, start=1.2, count=4, manager=manager)
        self.assertEqual(results, [None, None, None, 1])

    def test_wrong_person_at_same_position_does_not_inherit_id(self):
        manager = self.make_manager(same_camera_reconnect=True, reconnect_confirm_frames=4)
        self.feed(manager=manager)
        results = self.feed(local=2, feature=B, start=1, count=6, manager=manager)
        self.assertNotIn(1, results)
        self.assertEqual(results[-1], 2)  # Reject the implausible position candidate immediately.
        self.assertEqual(self.feed(local=2, feature=B, start=2, manager=manager)[-1], 2)
        np.testing.assert_array_equal(manager.global_tracks[1].features, [A])

    def test_same_camera_pose_change_reconnects_with_spatial_continuity(self):
        manager = self.make_manager(
            same_camera_reconnect=True,
            same_camera_reconnect_timeout=1800,
            same_camera_reconnect_distance=1.0,
            same_camera_reconnect_reid_threshold=.5,
            reconnect_confirm_threshold=.5,
            reconnect_confirm_frames=4,
            reconnect_confirm_ratio=.75)
        self.feed(manager=manager)
        changed = np.array([.54, math.sqrt(1 - .54 ** 2)], dtype=np.float32)
        moved_box = (-40, 20, 0, 100)  # center movement is below one box diagonal
        results = [manager.update_track(0, 2, changed, moved_box, .9, 10 + i * .1)
                   for i in range(6)]
        self.assertEqual(results[-1], 1)
        self.assertEqual(len(manager.global_tracks), 1)

    def test_same_camera_continuity_beats_slightly_closer_duplicate_gallery(self):
        manager = self.make_manager(
            threshold=.1,
            candidate_threshold=.45,
            same_camera_reconnect=True,
            same_camera_reconnect_timeout=1800,
            same_camera_reconnect_distance=1.0,
            same_camera_reconnect_reid_threshold=.5,
            same_camera_reconnect_reid_margin=.05,
            reconnect_confirm_threshold=.5,
            reconnect_confirm_frames=4,
            reconnect_confirm_ratio=.75)
        self.feed(manager=manager)  # GID 1 owns this position in camera 0.
        query = np.array([.65, math.sqrt(1-.65**2)], dtype=np.float32)
        # A different gallery is only .03 closer to the query, but has no
        # same-camera positional history here.
        angle = math.acos(.65) + math.acos(.68)
        duplicate = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
        self.feed(camera=1, feature=duplicate, start=10, manager=manager)
        self.assertEqual(len(manager.global_tracks), 2)
        results = [manager.update_track(0, 2, query, BOX, .9, 20+i*.1)
                   for i in range(6)]
        self.assertEqual(results[-1], 1)

    def test_old_position_not_used_for_reconnect(self):
        manager = self.make_manager(same_camera_reconnect=True)
        self.feed(manager=manager)
        self.assertEqual(self.feed(local=2, feature=B, start=20, manager=manager), [None, 2])

    def test_ambiguous_match_waits_then_gets_unique_id(self):
        manager = self.make_manager(pending_timeout=1)
        # Two identical-looking people co-occur in the same camera.
        tracks = [observation(1), observation(2)]
        manager.update_camera(0, tracks, 0)
        manager.update_camera(0, tracks, 0.1)
        self.assertEqual(self.feed(camera=1, start=1, manager=manager), [None, None])
        self.assertEqual(self.feed(camera=1, start=3, manager=manager), [None, 3])

    def test_invalid_features_do_not_create_an_identity(self):
        for feature in ([0, 0], [float('nan'), 0], [float('inf'), 1]):
            self.assertEqual(self.feed(feature=feature, count=5), [None] * 5)
        self.assertEqual(self.manager.global_tracks, {})

    def test_prediction_only_frames_do_not_confirm_candidates(self):
        self.feed()
        self.feed(camera=1, start=1)
        for now in range(2, 10):
            self.assertIsNone(self.manager.update_track(1, 1, None, BOX, 0, now))
        self.assertIsNone(self.manager.local_tracks[(1, 1)].global_id)

    def test_perfect_cross_camera_match_at_one_fps_does_not_time_out(self):
        manager = GlobalIDManager(same_camera_reconnect=False)
        for now in range(20):
            original = manager.update_track(0, 1, A, BOX, 0.9, now)
            matched = manager.update_track(1, 1, A, BOX, 0.9, now)
        self.assertEqual(original, 1)
        self.assertEqual(matched, 1)

    def test_candidate_reservation_expires_even_without_valid_features(self):
        manager = self.make_manager(pending_timeout=1)
        self.feed(manager=manager)
        self.feed(camera=1, start=1, manager=manager)
        self.assertEqual(manager.local_tracks[(1, 1)].candidate_id, 1)
        manager.update_track(1, 1, None, BOX, 0, 3)
        self.assertIsNone(manager.local_tracks[(1, 1)].candidate_id)
        self.assertEqual(self.feed(camera=1, local=2, start=4, count=4, manager=manager)[-1], 1)

    def test_camera_queue_delay_does_not_reset_other_camera_evidence(self):
        manager = GlobalIDManager(same_camera_reconnect=False)
        for now in range(20):
            manager.update_camera(0, [observation(1)], now + 20)
            result = manager.update_camera(1, [observation(1)], now)
        self.assertEqual(result[1], 1)

    def test_expired_candidate_samples_do_not_seed_a_different_person(self):
        manager = self.make_manager(pending_timeout=1)
        self.feed(manager=manager)
        self.feed(camera=1, start=1, manager=manager)
        self.assertEqual(self.feed(camera=1, feature=B, start=5, manager=manager)[-1], 2)
        np.testing.assert_array_equal(manager.global_tracks[2].features, [B])

    def test_rejected_position_can_confirm_an_existing_alternative(self):
        manager = self.make_manager(threshold=.1, same_camera_reconnect=True, reconnect_confirm_frames=4)
        self.feed(manager=manager)
        self.feed(camera=1, feature=B, start=0, manager=manager)
        # Appearance is outside strict acceptance, but supports a GID 1 reconnect.
        self.feed(local=2, feature=[.75, math.sqrt(1-.75**2)], start=1, manager=manager)
        self.feed(local=2, feature=B, start=1.2, count=4, manager=manager)
        self.assertIsNone(manager.local_tracks[(0, 2)].global_id)
        self.assertEqual(self.feed(local=2, feature=B, start=2, count=4, manager=manager)[-1], 2)
        self.assertEqual(len(manager.global_tracks), 2)

    def test_euclidean_gallery_updates_use_euclidean_distance(self):
        manager = self.make_manager(metric='euclidean', feature_update_max_distance=0.3)
        self.feed(manager=manager)
        # cosine distance=.1, Euclidean distance=sqrt(.2)>.3
        self.assertFalse(manager._add_global_feature(1, [0.9, math.sqrt(0.19)], 0.9))
        self.assertEqual(len(manager.global_tracks[1].features), 1)

    def test_quality_weighted_aggregation(self):
        expected = np.array([9.0, 1.0]) / np.linalg.norm([9.0, 1.0])
        np.testing.assert_allclose(mean_feature([A, B], [0.9, 0.1]), expected, rtol=1e-6)

    def test_impossible_feature_count_fails_fast(self):
        with self.assertRaises(ValueError):
            self.make_manager(min_features=100, track_feature_size=10)


class CameraTrackerTests(unittest.TestCase):
    def setUp(self):
        with patch.object(sys, 'argv', ['demo_stream.py', '--streams', 'placeholder']):
            self.args = parse_args()
        self.args.global_feature_min_blur = 0
        self.args.global_feature_min_box_height = 48
        self.args.global_feature_min_confidence = .4
        self.args.global_feature_min_track_hits = 2
        self.args.tracker_n_init = 2
        self.frame = np.zeros((160, 160, 3), dtype=np.uint8)

    def test_occlusion_blocks_features_but_not_tracks(self):
        self.assertEqual(reid_feature_quality(self.frame, BOX, 0.9, self.args, 5, [BOX]), 0)
        self.assertGreater(reid_feature_quality(self.frame, BOX, 0.9, self.args, 5), 0)

    def test_local_tracker_preserves_scores_and_handles_missed_frames(self):
        detector = SimpleNamespace(detect_image_with_scores=lambda image: ([[20, 20, 40, 80]], [0.9]))
        encoder = lambda frame, boxes, camera_id: np.array([A for _ in boxes])
        tracker = CameraTracker(0, detector, encoder, self.args)
        self.assertEqual(tracker.process(self.frame), [])
        tracks = tracker.process(self.frame)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0]['feature_quality'], 0.9)
        detector.detect_image_with_scores = lambda image: ([], [])
        predicted = tracker.process(self.frame)
        self.assertEqual(predicted, [])

    def test_invalid_embeddings_use_motion_without_contaminating_appearance(self):
        detector = SimpleNamespace(detect_image_with_scores=lambda image: ([[20, 20, 40, 80]], [0.9]))
        for invalid in ([0, 0], [float('nan'), 1], [float('inf'), 0], None):
            with self.subTest(feature=invalid):
                current = [invalid]
                encoder = lambda frame, boxes, camera_id: current
                tracker = CameraTracker(0, detector, encoder, self.args)
                for _ in range(6):
                    tracks = tracker.process(self.frame)
                self.assertEqual([t['local_id'] for t in tracks], [1])
                self.assertEqual(tracks[0]['feature_quality'], 0)
                self.assertEqual(tracker.tracker.metric.samples[1], [])
                current[:] = [A]
                for _ in range(3):
                    tracks = tracker.process(self.frame)
                self.assertEqual([t['local_id'] for t in tracks], [1])
                self.assertGreater(tracks[0]['feature_quality'], 0)
                self.assertTrue(np.all(np.isfinite(tracker.tracker.metric.samples[1])))

    def test_nested_same_appearance_track_is_suppressed(self):
        tracks = [
            observation(1, bbox=(10, 10, 50, 100)),
            observation(2, bbox=(20, 20, 45, 90)),
        ]
        kept = suppress_duplicate_tracks(tracks, .85, .15)
        self.assertEqual([track['local_id'] for track in kept], [1])

    def test_nested_different_people_are_not_suppressed(self):
        tracks = [
            observation(1, feature=A, bbox=(10, 10, 50, 100)),
            observation(2, feature=B, bbox=(20, 20, 45, 90)),
        ]
        kept = suppress_duplicate_tracks(tracks, .85, .15)
        self.assertEqual([track['local_id'] for track in kept], [1, 2])


if __name__ == '__main__':
    unittest.main()
