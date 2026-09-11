from __future__ import absolute_import, division, print_function

import numpy as np

from transreid_adapter import create_transreid_box_encoder


TRANSREID_INFERENCE_CAMERA_ID = 0


class TransReIDFeatureExtractorAdapter(object):
    backend = 'transreid'
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
        return self.extractor.extract(
            images,
            camera_id=TRANSREID_INFERENCE_CAMERA_ID,
            view_id=view_id
        )

    def extract_numpy(self, images, camera_id=None, view_id=None):
        return self.extract(images, camera_id=camera_id, view_id=view_id).numpy().astype(np.float32)


class TransReIDBoxEncoderAdapter(object):
    backend = 'transreid'
    uses_camera_id = False
    uses_view_id = False

    def __init__(
        self,
        variant='msmt17',
        weight_path=None,
        repo_dir=None,
        assets_root=None,
        batch_size=8,
        device=None,
        download=True
    ):
        self.encoder = create_transreid_box_encoder(
            variant=variant,
            weight_path=weight_path,
            repo_dir=repo_dir,
            assets_root=assets_root,
            batch_size=batch_size,
            device=device,
            download=download
        )
        self.extractor = TransReIDFeatureExtractorAdapter(self.encoder.extractor)
        self.feature_dim = self.extractor.feature_dim
        print('TransReID inference cam_label:', TRANSREID_INFERENCE_CAMERA_ID)

    def __call__(self, image, boxes, camera_id=None, view_id=None):
        return self.encoder(
            image,
            boxes,
            camera_id=TRANSREID_INFERENCE_CAMERA_ID,
            view_id=view_id
        )
