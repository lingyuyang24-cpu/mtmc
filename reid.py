from __future__ import absolute_import, division, print_function

from torchreid import metrics

from torch_box_encoder import TorchReIDFeatureExtractor


class REID(object):
    def __init__(
        self,
        model_name='resnet50',
        weights_path='model_data/models/model.pth',
        batch_size=32,
        use_gpu=None,
        extractor=None,
        dist_metric='cosine'
    ):
        self.extractor = extractor
        if self.extractor is None:
            self.extractor = TorchReIDFeatureExtractor(
                model_name=model_name,
                weights_path=weights_path,
                batch_size=batch_size,
                use_gpu=use_gpu
            )
        self.dist_metric = dist_metric

    def _features(self, imgs):
        return self.extractor.extract(imgs)

    def compute_distance(self, qf, gf):
        distmat = metrics.compute_distance_matrix(qf, gf, self.dist_metric)
        return distmat.numpy()


if __name__ == '__main__':
    reid = REID()
