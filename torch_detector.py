# ! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import os

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F


class UltralyticsYOLOPersonDetector(object):
    """Ultralytics YOLO person detector with the old YOLO wrapper output shape."""

    COCO_PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_name='yolov8n.pt',
        score_threshold=0.6,
        device=None,
        imgsz=640
    ):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                'Ultralytics YOLO is not installed. Run: python -m pip install ultralytics'
            )

        self.model_name = model_name
        self.score_threshold = score_threshold
        if not isinstance(imgsz, int) or imgsz < 32:
            raise ValueError('Detector image size must be an integer >= 32.')
        self.imgsz = imgsz
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(model_name)

    def __repr__(self):
        return '{}(model_name={!r}, device={!r})'.format(
            self.__class__.__name__, self.model_name, self.device
        )

    def detect_image_with_scores(self, image):
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))

        image = image.convert('RGB')
        # Ultralytics treats PIL as RGB and numpy arrays as BGR.
        # Keep the PIL type so that its loader performs the correct conversion.
        results = self.model.predict(
            source=image,
            conf=self.score_threshold,
            classes=[self.COCO_PERSON_CLASS_ID],
            device=self.device,
            verbose=False,
            imgsz=self.imgsz
        )

        detections = []
        detection_scores = []
        if not results:
            return detections, detection_scores

        boxes = results[0].boxes
        if boxes is None:
            return detections, detection_scores

        xyxy = boxes.xyxy.detach().cpu().numpy()
        scores = boxes.conf.detach().cpu().numpy()
        labels = boxes.cls.detach().cpu().numpy()

        for box, score, label in zip(xyxy, scores, labels):
            if score < self.score_threshold:
                continue
            if int(label) != self.COCO_PERSON_CLASS_ID:
                continue

            x1, y1, x2, y2 = box
            x1 = max(0, int(round(x1)))
            y1 = max(0, int(round(y1)))
            x2 = min(image.width - 1, int(round(x2)))
            y2 = min(image.height - 1, int(round(y2)))
            width = max(0, x2 - x1)
            height = max(0, y2 - y1)
            if width > 0 and height > 0:
                detections.append([x1, y1, width, height])
                detection_scores.append(float(score))

        return detections, detection_scores

    def detect_image(self, image):
        detections, _ = self.detect_image_with_scores(image)
        return detections


class TorchVisionPersonDetector(object):
    """Torchvision fallback detector with the same output shape as the old YOLO wrappers."""

    COCO_PERSON_CLASS_ID = 1

    def __init__(
        self,
        model_name='fasterrcnn_resnet50_fpn',
        weights='default',
        score_threshold=0.6,
        device=None
    ):
        self.model_name = model_name
        self.weights = weights
        self.score_threshold = score_threshold
        self.device = torch.device(
            device or ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.model = self._load_model(model_name, weights)
        self.model.to(self.device)
        self.model.eval()

    def __repr__(self):
        return '{}(model_name={!r}, weights={!r}, device={!r})'.format(
            self.__class__.__name__, self.model_name, self.weights, str(self.device)
        )

    def _load_model(self, model_name, weights):
        if weights in ('none', 'random', '', None, False):
            return self._build_model_without_weights(model_name)

        if weights not in ('default', 'DEFAULT', True):
            weights_path = str(weights)
            if not os.path.isfile(weights_path):
                raise FileNotFoundError('Detector weights not found: {}'.format(weights_path))
            model = self._build_model_without_weights(model_name)
            checkpoint = torch.load(weights_path, map_location='cpu')
            state_dict = checkpoint.get('state_dict', checkpoint)
            if 'model' in state_dict:
                state_dict = state_dict['model']
            model.load_state_dict(state_dict, strict=False)
            return model

        try:
            import torchvision.models as models
            if hasattr(models, 'get_model'):
                return models.get_model(model_name, weights='DEFAULT')
        except Exception as exc:
            first_error = exc
        else:
            first_error = None

        try:
            import torchvision.models.detection as detection
            builder = getattr(detection, model_name)
        except AttributeError:
            raise ValueError('Unknown torchvision detection model: {}'.format(model_name))

        try:
            return builder(weights='DEFAULT')
        except TypeError:
            return builder(pretrained=True)
        except Exception as exc:
            if first_error is not None:
                raise RuntimeError(
                    'Could not load default weights for {}. First error: {}. '
                    'Second error: {}'.format(model_name, first_error, exc)
                )
            raise

    def _build_model_without_weights(self, model_name):
        try:
            import torchvision.models as models
            if hasattr(models, 'get_model'):
                return models.get_model(
                    model_name,
                    weights=None,
                    weights_backbone=None
                )
        except TypeError:
            pass

        import torchvision.models.detection as detection
        try:
            builder = getattr(detection, model_name)
        except AttributeError:
            raise ValueError('Unknown torchvision detection model: {}'.format(model_name))

        try:
            return builder(weights=None, weights_backbone=None)
        except TypeError:
            return builder(pretrained=False)

    def detect_image_with_scores(self, image):
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))

        image = image.convert('RGB')
        tensor = F.to_tensor(image).to(self.device)

        with torch.no_grad():
            prediction = self.model([tensor])[0]

        boxes = prediction['boxes'].detach().cpu().numpy()
        scores = prediction['scores'].detach().cpu().numpy()
        labels = prediction['labels'].detach().cpu().numpy()

        detections = []
        detection_scores = []
        for box, score, label in zip(boxes, scores, labels):
            if score < self.score_threshold:
                continue
            if int(label) != self.COCO_PERSON_CLASS_ID:
                continue

            x1, y1, x2, y2 = box
            x1 = max(0, int(round(x1)))
            y1 = max(0, int(round(y1)))
            x2 = min(image.width - 1, int(round(x2)))
            y2 = min(image.height - 1, int(round(y2)))
            width = max(0, x2 - x1)
            height = max(0, y2 - y1)
            if width > 0 and height > 0:
                detections.append([x1, y1, width, height])
                detection_scores.append(float(score))

        return detections, detection_scores

    def detect_image(self, image):
        detections, _ = self.detect_image_with_scores(image)
        return detections


def _looks_like_yolo_model(model_name):
    model_name = str(model_name).lower()
    return (
        model_name.startswith('yolo') or
        os.path.basename(model_name).startswith('yolo') or
        model_name.endswith('.pt')
    )


def build_person_detector(
    model_name='yolov8n.pt',
    backend='auto',
    weights='default',
    score_threshold=0.6,
    device=None,
    imgsz=640
):
    if backend not in ('auto', 'ultralytics', 'torchvision'):
        raise ValueError('Unknown detector backend: {}'.format(backend))

    if backend == 'ultralytics' or (
        backend == 'auto' and _looks_like_yolo_model(model_name)
    ):
        if weights not in ('default', 'DEFAULT', None, '', True):
            model_name = weights
        return UltralyticsYOLOPersonDetector(
            model_name=model_name,
            score_threshold=score_threshold,
            device=device,
            imgsz=imgsz
        )

    return TorchVisionPersonDetector(
        model_name=model_name,
        weights=weights,
        score_threshold=score_threshold,
        device=device
    )
