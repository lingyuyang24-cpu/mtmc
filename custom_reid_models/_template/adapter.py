from __future__ import absolute_import, division, print_function


class ReIDExtractor(object):
    """Template for custom ReID models.

    Copy this folder, replace the model loading and preprocessing code, and make
    extract(...) return a numpy array or torch tensor with shape [N, feature_dim].
    """

    feature_dim = 512
    uses_camera_id = False
    uses_view_id = False

    def __init__(
        self,
        model_dir,
        weights_path=None,
        config_path=None,
        config=None,
        model_name=None,
        batch_size=32,
        device=None,
        normalize=True
    ):
        raise NotImplementedError(
            'Copy custom_reid_models/_template to a new folder and implement ReIDExtractor.'
        )

    def extract(self, images, camera_id=None, view_id=None):
        raise NotImplementedError
