import importlib.util
import unittest
from types import SimpleNamespace

import numpy as np
from PIL import Image


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('torchvision'),
                     'Detector adapter test needs torch and torchvision, but no weights.')
class DetectorColorTests(unittest.TestCase):
    def test_rgb_pil_is_not_passed_as_a_bgr_numpy_array(self):
        from torch_detector import UltralyticsYOLOPersonDetector
        captured = {}
        def predict(**kwargs):
            captured.update(kwargs)
            return []
        detector = UltralyticsYOLOPersonDetector.__new__(UltralyticsYOLOPersonDetector)
        detector.model = SimpleNamespace(predict=predict)
        detector.device = 'cpu'
        detector.score_threshold = 0.3
        detector.imgsz = 1280
        red = Image.new('RGB', (8, 8), (255, 0, 0))
        self.assertEqual(detector.detect_image_with_scores(red), ([], []))
        self.assertIsInstance(captured['source'], Image.Image)
        self.assertEqual(captured['imgsz'], 1280)
        np.testing.assert_array_equal(np.asarray(captured['source']), np.asarray(red))


if __name__ == '__main__':
    unittest.main()
