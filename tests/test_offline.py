"""Offline regressions: real Deep SORT/video I/O, mocked detector and encoder.

Run: .venv/bin/python -m unittest discover -s tests -p test_offline.py -v
Dependencies: numpy, scipy, opencv-python (or headless), Pillow. No torch/weights.
"""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import demo
from offline_association import RepresentativeGallery, Tracklet, feature_distance, fuse_tracklets


def vector(degrees):
    radians = np.deg2rad(degrees)
    return np.array([np.cos(radians), np.sin(radians)])


def track(camera, local, output_id, start=0, end=10, degrees=0, count=3, qualities=None):
    return Tracklet((camera, local), output_id, start, end,
                    np.tile(vector(degrees), (count, 1)), qualities)


class FusionTests(unittest.TestCase):
    def fuse(self, tracks, **kwargs):
        kwargs.setdefault('min_frames', 2)
        return fuse_tracklets(tracks, **kwargs)

    def test_short_tracks_keep_unique_ids_and_are_not_merge_targets(self):
        tracks = [track(0, 1, 1, count=1), track(1, 1, 2), track(2, 1, 3, count=1)]
        mapping, groups = self.fuse(tracks)
        self.assertEqual(mapping, {(0, 1): 1, (1, 1): 2, (2, 1): 3})
        self.assertEqual(len(groups), 3)

    def test_empty_and_invalid_features_keep_unique_ids(self):
        tracks = [track(0, 1, 1), track(1, 1, 2, count=0), track(2, 1, 3)]
        tracks[2].features[:] = np.nan
        mapping, _ = self.fuse(tracks)
        self.assertEqual(list(mapping.values()), [1, 2, 3])

    def test_full_spans_reject_later_same_camera_coexistence(self):
        # Track 1 may be unmatched when track 2 starts, but returns at frame 20.
        mapping, _ = self.fuse([track(0, 1, 1, 0, 20), track(0, 2, 2, 10, 30)])
        self.assertNotEqual(mapping[(0, 1)], mapping[(0, 2)])

    def test_conflict_with_nonfounder_group_member_blocks_merge(self):
        tracks = [track(0, 1, 1, 0, 5), track(1, 1, 2, 0, 20), track(1, 2, 3, 10, 30)]
        mapping, groups = self.fuse(tracks)
        self.assertEqual(mapping[(0, 1)], mapping[(1, 1)])
        self.assertNotEqual(mapping[(1, 1)], mapping[(1, 2)])
        self.assertEqual(len(groups[1].members), 2)

    def test_same_camera_nonoverlap_can_merge_but_endpoints_conflict(self):
        for start, expected in ((5, 2), (6, 1)):
            with self.subTest(start=start):
                mapping, _ = self.fuse([track(0, 1, 1, 0, 5), track(0, 2, 2, start, 12)])
                self.assertEqual(mapping[(0, 2)], expected)

    def test_cross_camera_overlap_is_allowed(self):
        mapping, _ = self.fuse([track(0, 1, 1), track(1, 1, 2)])
        self.assertEqual(mapping, {(0, 1): 1, (1, 1): 1})

    def test_merge_updates_gallery_for_later_candidates(self):
        tracks = [track(0, 1, 1, degrees=0), track(1, 1, 2, degrees=30),
                  track(2, 1, 3, degrees=55)]
        self.assertGreater(feature_distance([vector(0)], [vector(55)])[0, 0], .3)
        mapping, groups = self.fuse(tracks, threshold=.3)
        self.assertEqual(set(mapping.values()), {1})
        self.assertEqual(groups[1].gallery.sample_count, 9)

    def test_margin_preserves_ambiguous_track(self):
        tracks = [track(0, 1, 1, degrees=-20), track(0, 2, 2, degrees=20),
                  track(1, 1, 3, degrees=0)]
        mapping, _ = self.fuse(tracks, threshold=.2, margin=.05)
        self.assertEqual(mapping[(1, 1)], 3)
        mapping, _ = self.fuse(tracks, threshold=.2, margin=0)
        self.assertIn(mapping[(1, 1)], (1, 2))

    def test_confidence_weights_reduce_poor_sample_influence(self):
        source = track(0, 1, 1, count=2, qualities=[.99, .01])
        source.features = np.array([vector(0), vector(90)])
        query = track(1, 1, 2)
        weighted, _ = self.fuse([source, query], threshold=.05)
        self.assertEqual(weighted[(1, 1)], 1)
        source.qualities = None
        uniform, _ = self.fuse([source, query], threshold=.05)
        self.assertEqual(uniform[(1, 1)], 2)

    def test_gallery_is_bounded_and_preserves_support_after_merge(self):
        gallery = RepresentativeGallery([vector(i) for i in range(90)], limit=4)
        other = RepresentativeGallery([vector(i) for i in range(90, 180)], limit=4)
        gallery.merge(other)
        self.assertLessEqual(len(gallery.features), 4)
        self.assertEqual(gallery.sample_count, 180)
        self.assertAlmostEqual(gallery.weights.sum(), 180)
        self.assertTrue(np.isfinite(gallery.distance(other)))

    def test_euclidean_is_ordinary_distance_on_normalized_features(self):
        self.assertAlmostEqual(feature_distance([[3, 0]], [[0, 7]], 'euclidean')[0, 0], np.sqrt(2))
        self.assertAlmostEqual(feature_distance([[3, 0]], [[0, 7]], 'cosine')[0, 0], 1)
        self.assertAlmostEqual(feature_distance([[3, 0]], [[8, 0]], 'euclidean')[0, 0], 0)
        tracks = [track(0, 1, 1, degrees=0), track(1, 1, 2, degrees=90)]
        mapping, _ = self.fuse(tracks, metric='euclidean', threshold=1.5)
        self.assertEqual(mapping[(1, 1)], 1)
        mapping, _ = self.fuse(tracks, metric='euclidean', threshold=1.3)
        self.assertEqual(mapping[(1, 1)], 2)

    def test_both_metrics_reject_zero_and_nonfinite_embeddings(self):
        for metric in ('cosine', 'euclidean'):
            with self.subTest(metric=metric):
                gallery = RepresentativeGallery([[0, 0], [np.nan, 1], [np.inf, 0], [1, 0]], metric=metric)
                self.assertEqual(gallery.sample_count, 1)
                self.assertTrue(np.isinf(feature_distance([[0, 0]], [[1, 0]], metric)[0, 0]))
                a, b = track(0, 1, 1), track(1, 1, 2)
                a.features[:] = b.features[:] = 0
                mapping, _ = self.fuse([a, b], metric=metric)
                self.assertEqual(list(mapping.values()), [1, 2])

    def test_order_is_deterministic_and_duplicate_ids_rejected(self):
        tracks = [track(0, 1, 1), track(1, 1, 2)]
        self.assertEqual(self.fuse(tracks)[0], self.fuse(reversed(tracks))[0])
        with self.assertRaises(ValueError):
            self.fuse([tracks[0], tracks[0]])
        with self.assertRaises(ValueError):
            RepresentativeGallery([[1, 0]], qualities=[])


class DetectionAndSampleTests(unittest.TestCase):
    def test_import_does_not_parse_arguments_or_import_neural_backends(self):
        result = subprocess.run(
            [sys.executable, '-c',
             "import sys; sys.argv=['demo.py','--not-a-real-option']; "
             "sys.modules['torch']=None; sys.modules['torchvision']=None; import demo; "
             "assert 'torch_detector' not in sys.modules; assert 'torchreid' not in sys.modules; "
             "assert 'reid_backends' not in sys.modules"],
            cwd=str(ROOT), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_detector_mode_preserves_overlap_and_real_scores(self):
        args = demo.parser.parse_args(['--videos', 'mock.avi'])
        self.assertEqual(args.post_nms, 'detector')
        detector = mock.Mock()
        detector.detect_image_with_scores.return_value = ([[2, 2, 8, 8], [2, 2, 8, 8]], [.23, .87])
        encoder = mock.Mock(return_value=np.array([[1, 0], [0, 1]]))
        detections = demo.prepare_detections(detector, encoder, np.zeros((20, 20, 3), np.uint8), 4)
        self.assertEqual([d.confidence for d in detections], [.23, .87])
        self.assertEqual(len(encoder.call_args.args[1]), 2)
        self.assertEqual(encoder.call_args.kwargs['camera_id'], 4)

    def test_nms_selects_before_encoding_and_keeps_high_score(self):
        detector = mock.Mock()
        detector.detect_image_with_scores.return_value = ([[2, 2, 8, 8], [2, 2, 8, 8]], [.23, .87])
        encoder = mock.Mock(return_value=np.array([[1, 0]]))
        detections = demo.prepare_detections(
            detector, encoder, np.zeros((20, 20, 3), np.uint8), 0, post_nms='nms')
        self.assertEqual(len(encoder.call_args.args[1]), 1)
        self.assertEqual([d.confidence for d in detections], [.87])

    def test_empty_detections_skip_encoder_and_mismatched_scores_raise(self):
        detector, encoder = mock.Mock(), mock.Mock()
        detector.detect_image_with_scores.return_value = ([], [])
        self.assertEqual(demo.prepare_detections(detector, encoder, np.zeros((20, 20, 3), np.uint8), 0), [])
        encoder.assert_not_called()
        detector.detect_image_with_scores.return_value = ([[1, 1, 2, 2]], [])
        with self.assertRaises(ValueError):
            demo.prepare_detections(detector, encoder, np.zeros((20, 20, 3), np.uint8), 0)

    def test_crop_clips_and_owns_clean_pixels(self):
        frame = np.full((12, 12, 3), 37, np.uint8)
        sample = demo.make_track_sample(frame, [-2, -2, 6, 6], 3, .73)
        self.assertEqual(sample['image'].shape, (6, 6, 3))
        self.assertFalse(np.shares_memory(sample['image'], frame))
        frame[:] = 255
        self.assertTrue(np.all(sample['image'] == 37))
        self.assertEqual(sample['camera_id'], 3)
        self.assertEqual(sample['quality'], .73)
        self.assertIsNone(demo.make_track_sample(frame, [10, 10, 3, 3], 3))
        self.assertIsNone(demo.make_track_sample(frame, [0, 0, 3, 3], 3, float('nan')))

    def test_camera_aware_extraction_preserves_sample_order(self):
        extractor = mock.Mock(uses_camera_id=True, feature_dim=2)
        extractor.extract.side_effect = lambda images, camera_id: np.tile([camera_id + 1, 1], (len(images), 1))
        samples = [{'camera_id': camera, 'image': np.zeros((2, 2, 3), np.uint8)} for camera in (2, 0, 2)]
        np.testing.assert_array_equal(demo.extract_track_features(extractor, samples), [[3, 1], [1, 1], [3, 1]])


class MockExtractor:
    uses_camera_id = True
    feature_dim = 2
    batch_size = 7

    def __init__(self):
        self.images = []

    def extract(self, images, camera_id=None):
        self.images.extend(image.copy() for image in images)
        return np.tile(vector(camera_id * 90), (len(images), 1))


class MockEncoder:
    def __init__(self, invalid=False):
        self.extractor = MockExtractor()
        self.shapes = []
        self.invalid = invalid

    def __call__(self, frame, boxes, camera_id=None):
        self.shapes.append((camera_id, frame.shape))
        return np.tile([0, 0] if self.invalid else vector(camera_id * 90), (len(boxes), 1))


class MockDetector:
    def __init__(self, miss_last=False, overlapping=False):
        self.calls = 0
        self.miss_last = miss_last
        self.overlapping = overlapping

    def detect_image_with_scores(self, image):
        self.calls += 1
        if self.miss_last and self.calls % 4 == 0:
            return [], []
        boxes, scores = [[12, 8, 24, 32]], [.73]
        if self.overlapping:
            boxes.append([20, 8, 24, 32])
            scores.append(.81)
        return boxes, scores


class VideoIntegrationTests(unittest.TestCase):
    def make_video(self, directory, name, size, count=4):
        path = Path(directory) / name
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, size)
        self.assertTrue(writer.isOpened(), 'OpenCV MJPG encoder is required for integration tests.')
        for _ in range(count):
            writer.write(np.full((size[1], size[0], 3), 40, np.uint8))
        writer.release()
        return str(path)

    def read_video(self, path):
        capture = cv2.VideoCapture(str(path))
        frames = []
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(frame)
        finally:
            capture.release()
        return frames

    def test_independent_camera_ids_mixed_sizes_and_all_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            videos = [self.make_video(directory, 'a.avi', (64, 64)),
                      self.make_video(directory, 'b.avi', (96, 80))]
            args = demo.parser.parse_args(['--videos', *videos, '--tracker-n-init', '2',
                                          '--min-reid-frames', '10', '--side-by-side-scale', '1'])
            output = Path(directory) / 'output'
            trackers = []
            real_tracker = demo.Tracker

            def create_tracker(*a, **kw):
                tracker = real_tracker(*a, **kw)
                trackers.append(tracker)
                return tracker

            encoder = MockEncoder()
            with mock.patch.object(demo, 'Tracker', side_effect=create_tracker), contextlib.redirect_stdout(io.StringIO()):
                mapping = demo.main(MockDetector(), args, encoder, output)
            self.assertEqual(mapping, {(0, 1): 1, (1, 1): 2})
            self.assertEqual(len(trackers), 2)
            self.assertIsNot(trackers[0].metric, trackers[1].metric)
            self.assertEqual([tracker.tracks[0].age for tracker in trackers], [4, 4])
            self.assertEqual([tracker.tracks[0].track_id for tracker in trackers], [1, 1])
            self.assertEqual([tracker.tracks[0].last_confidence for tracker in trackers], [.73, .73])
            self.assertAlmostEqual(trackers[0].metric.samples[1][-1][0], 1)
            self.assertAlmostEqual(trackers[1].metric.samples[1][-1][1], 1)
            document = json.loads((output / 'id_mapping.json').read_text())
            self.assertEqual([(t['camera_id'], t['local_id'], t['tracking_id'], t['global_id'])
                              for t in document['tracks']], [(0, 1, 1, 1), (1, 1, 2, 2)])
            self.assertEqual([t['start_frame'] for t in document['tracks']], [0, 0])
            rows = [line.split(',') for line in (output / 'tracking.txt').read_text().splitlines()]
            self.assertEqual({int(row[1]) for row in rows}, {1, 2})
            self.assertEqual([(int(row[-2]), int(row[-1])) for row in rows if row[1] == '2'], [(96, 80)] * 3)
            self.assertIn((1, (80, 96, 3)), encoder.shapes)
            self.assertTrue(all(image.shape == (32, 24, 3) for image in encoder.extractor.images))
            self.assertEqual(encoder.extractor.batch_size, 7)
            for name in ('tracking.avi', 'Complete.avi'):
                frames = self.read_video(output / name)
                self.assertEqual(len(frames), 8, name)
                self.assertTrue(all(frame.shape == (64, 64, 3) for frame in frames))
            for name in ('tracking_side_by_side.avi', 'Complete_side_by_side.avi'):
                frames = self.read_video(output / name)
                self.assertEqual(len(frames), 4, name)
                self.assertTrue(all(frame.shape == (64, 128, 3) for frame in frames))
            self.assertFalse((output / 'allVideos.avi').exists())

    def test_overlapping_tracks_crops_are_clean_and_predictions_are_not_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            video = self.make_video(directory, 'overlap.avi', (64, 64))
            args = demo.parser.parse_args(['--videos', video, '--tracker-n-init', '2',
                                          '--all', 'false', '--side-by-side', 'false'])
            output = Path(directory) / 'output'
            encoder = MockEncoder()

            def contaminate_drawing(track_id, frame, *unused):
                frame[:] = 255

            with mock.patch.object(demo, 'cv2_addBox', side_effect=contaminate_drawing), contextlib.redirect_stdout(io.StringIO()):
                demo.main(MockDetector(miss_last=True, overlapping=True), args, encoder, output)
            document = json.loads((output / 'id_mapping.json').read_text())
            self.assertEqual(len(document['tracks']), 2)
            self.assertEqual([t['reid_samples'] for t in document['tracks']], [2, 2])
            self.assertEqual(len(encoder.extractor.images), 4)
            self.assertTrue(all(np.max(image) < 50 for image in encoder.extractor.images))
            rows = (output / 'tracking.txt').read_text().splitlines()
            self.assertEqual(len(rows), 6)  # The predicted final frame is still rendered/logged.

    def test_invalid_embeddings_do_not_store_reid_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            video = self.make_video(directory, 'invalid.avi', (64, 64))
            args = demo.parser.parse_args(['--videos', video, '--tracker-n-init', '2',
                                          '--all', 'false', '--side-by-side', 'false'])
            output = Path(directory) / 'output'
            encoder = MockEncoder(invalid=True)
            with contextlib.redirect_stdout(io.StringIO()):
                mapping = demo.main(MockDetector(), args, encoder, output)
            self.assertEqual(mapping, {(0, 1): 1})
            self.assertEqual(encoder.extractor.images, [])
            document = json.loads((output / 'id_mapping.json').read_text())
            self.assertEqual(document['tracks'][0]['reid_samples'], 0)


if __name__ == '__main__':
    unittest.main()
