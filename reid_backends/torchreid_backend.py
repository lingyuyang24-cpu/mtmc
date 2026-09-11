from __future__ import absolute_import, division, print_function

import numpy as np

from torch_box_encoder import create_box_encoder


class TorchReIDFeatureExtractorAdapter(object):
    backend = 'torchreid'
    uses_camera_id = False
    uses_view_id = False

    def __init__(self, extractor):
        self.extractor = extractor
        self.feature_dim = getattr(extractor, 'feature_dim', 0)

    @property
    def batch_size(self):
        return getattr(self.extractor, 'batch_size', None)

    @batch_size.setter
    def batch_size(self, value):
        if hasattr(self.extractor, 'batch_size'):
            self.extractor.batch_size = value

    def extract(self, images, camera_id=None, view_id=None):
        return self.extractor.extract(images)

    def extract_numpy(self, images, camera_id=None, view_id=None):
        return self.extract(images).numpy().astype(np.float32)


class TorchReIDBoxEncoderAdapter(object):
    backend = 'torchreid'
    uses_camera_id = False
    uses_view_id = False

    def __init__(
        self,
        model_name='resnet50',
        weights_path='model_data/models/model.pth',
        batch_size=32,
        use_gpu=None
    ):
        self.encoder = create_box_encoder(
            model_name=model_name,
            weights_path=weights_path,
            batch_size=batch_size,
            use_gpu=use_gpu
        )
        self.extractor = TorchReIDFeatureExtractorAdapter(self.encoder.extractor)
        self.feature_dim = self.extractor.feature_dim

    def __call__(self, image, boxes, camera_id=None, view_id=None):
        return self.encoder(image, boxes)
