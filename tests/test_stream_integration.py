"""Real file decoding, Deep SORT, online association and video/log output.

Only the detector and encoder are synthetic, making identity ground truth exact
and keeping this test independent of GPU availability and downloaded weights.
"""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from demo_stream import main


class FixedDetector:
    def detect_image_with_scores(self, image):
        return [[20, 20, 40, 80]], [0.95]


class FixedEncoder:
    def __call__(self, frame, boxes, camera_id=None):
        return np.array([[1.0, 0.0] for _ in boxes], dtype=np.float32)


class StreamIntegrationTests(unittest.TestCase):
    def test_two_file_streams_share_confirmed_gid_and_save_output(self):
        with tempfile.TemporaryDirectory(prefix='mtmc-stream-test-') as root:
            root = Path(root)
            paths = [root / 'camera0.avi', root / 'camera1.avi']
            for camera, path in enumerate(paths):
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 30, (160, 160))
                self.assertTrue(writer.isOpened())
                for _ in range(12):
                    frame = np.full((160, 160, 3), 30 + camera * 40, dtype=np.uint8)
                    writer.write(frame)
                writer.release()

            output = root / 'result.avi'
            log = root / 'tracks.jsonl'
            argv = ['--streams', *(str(p) for p in paths), '--stream-mode', 'queue',
                    '--stream-queue-size', '3', '--display', 'false',
                    '--output', str(output), '--track-log', str(log),
                    '--global-delay', '0', '--global-min-features', '2',
                    '--global-feature-min-track-hits', '2', '--tracker-n-init', '2',
                    '--global-feature-aggregate-frames', '2', '--global-confirm-windows', '2',
                    '--global-feature-min-blur', '0', '--global-feature-min-box-height', '48']
            with contextlib.redirect_stdout(io.StringIO()):
                main(argv, detector=FixedDetector(), encoder=FixedEncoder())
            rows = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual({row['camera_id'] for row in rows}, {0, 1})
            for camera in (0, 1):
                camera_rows = [row for row in rows if row['camera_id'] == camera]
                self.assertEqual([row['frame'] for row in camera_rows], list(range(2, 13)))
                self.assertEqual({row['local_id'] for row in camera_rows}, {1})
                self.assertEqual(camera_rows[-1]['global_id'], 1)
                self.assertTrue(any(row['global_id'] is None for row in camera_rows))
            cap = cv2.VideoCapture(str(output))
            self.assertTrue(cap.isOpened())
            ok, frame = cap.read()
            cap.release()
            self.assertTrue(ok)
            self.assertEqual(frame.shape[:2], (160, 320))


if __name__ == '__main__':
    unittest.main()
