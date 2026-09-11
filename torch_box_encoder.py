# ! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import os
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import torchreid
from torchreid.data.transforms import build_transforms


def extract_image_patch(image, bbox):
    """Extract a BGR image patch from an ``(x, y, width, height)`` box."""
    image = np.asarray(image)
    bbox = np.asarray(bbox, dtype=np.float32).copy()
    bbox[2:] += bbox[:2]

    x1, y1, x2, y2 = bbox.astype(np.int32)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)

    if x1 >= x2 or y1 >= y2:
        return None

    return image[y1:y2, x1:x2]


class TorchReIDFeatureExtractor(object):
    """Torchreid feature extractor used by both Deep SORT and final ReID fusion."""

    def __init__(
        self,
        model_name='resnet50',
        weights_path='model_data/models/model.pth',
        batch_size=32,
        use_gpu=None,
        pretrained=True,
        normalize=True
    ):
        self.model_name = model_name
        self.weights_path = weights_path
        self.batch_size = batch_size
        self.use_gpu = torch.cuda.is_available() if use_gpu is None else use_gpu
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        self.normalize = normalize

        self.model = self._build_model(pretrained=pretrained)
        self._load_weights(weights_path)
        self.model.to(self.device)
        self.model.eval()
        self.feature_dim = getattr(self.model, 'feature_dim', 0)

        _, self.transform_te = build_transforms(
            height=256,
            width=128,
            random_erase=False,
            color_jitter=False,
            color_aug=False
        )

    def _build_model(self, pretrained=True):
        try:
            return torchreid.models.build_model(
                name=self.model_name,
                num_classes=1,
                loss='softmax',
                pretrained=pretrained,
                use_gpu=self.use_gpu
            )
        except Exception as exc:
            if not pretrained:
                raise
            warnings.warn(
                'Could not load ImageNet-pretrained {} weights ({}). '
                'Falling back to randomly initialized weights.'.format(
                    self.model_name, exc
                )
            )
            return torchreid.models.build_model(
                name=self.model_name,
                num_classes=1,
                loss='softmax',
                pretrained=False,
                use_gpu=self.use_gpu
            )

    def _load_weights(self, weights_path):
        if not weights_path:
            return

        if os.path.isfile(weights_path):
            torchreid.utils.load_pretrained_weights(self.model, weights_path)
            return

        warnings.warn(
            'ReID weights not found at "{}"; continuing with the model '
            'initialization weights.'.format(weights_path)
        )

    def _to_pil_rgb(self, image):
        if isinstance(image, Image.Image):
            return image.convert('RGB')

        image = np.asarray(image)
        if image.size == 0:
            raise ValueError('empty image patch')

        if image.ndim == 2:
            return Image.fromarray(image.astype('uint8')).convert('RGB')

        # Frames and crops in this project come from OpenCV, so they are BGR.
        image = image.astype('uint8')
        if image.shape[2] >= 3:
            image = image[:, :, :3][:, :, ::-1]
        return Image.fromarray(image).convert('RGB')

    def extract(self, images):
        tensors = []
        for image in images:
            try:
                tensors.append(self.transform_te(self._to_pil_rgb(image)))
            except ValueError:
                continue

        if not tensors:
            return torch.empty((0, self.feature_dim), dtype=torch.float32)

        features = []
        with torch.no_grad():
            for start in range(0, len(tensors), self.batch_size):
                batch = torch.stack(tensors[start:start + self.batch_size]).to(self.device)
                batch_features = self.model(batch)
                if isinstance(batch_features, (tuple, list)):
                    batch_features = batch_features[-1]
                if self.normalize:
                    batch_features = F.normalize(batch_features, p=2, dim=1)
                features.append(batch_features.detach().cpu())

        return torch.cat(features, 0)

    def extract_numpy(self, images):
        return self.extract(images).numpy().astype(np.float32)


class TorchBoxEncoder(object):
    def __init__(
        self,
        model_name='resnet50',
        weights_path='model_data/models/model.pth',
        batch_size=32,
        use_gpu=None
    ):
        self.extractor = TorchReIDFeatureExtractor(
            model_name=model_name,
            weights_path=weights_path,
            batch_size=batch_size,
            use_gpu=use_gpu
        )
        self.feature_dim = self.extractor.feature_dim

    def __call__(self, image, boxes):
        patches = []
        for box in boxes:
            patch = extract_image_patch(image, box)
            if patch is None:
                patch = np.zeros((256, 128, 3), dtype=np.uint8)
            patches.append(patch)

        if not patches:
            return np.empty((0, self.feature_dim), dtype=np.float32)

        return self.extractor.extract_numpy(patches)


def create_box_encoder(
    model_name='resnet50',
    weights_path='model_data/models/model.pth',
    batch_size=32,
    use_gpu=None
):
    return TorchBoxEncoder(
        model_name=model_name,
        weights_path=weights_path,
        batch_size=batch_size,
        use_gpu=use_gpu
    )
