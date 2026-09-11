# ! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import argparse
import importlib
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


TRANSREID_REPO_URL = 'https://github.com/damo-cv/TransReID/archive/refs/heads/main.zip'

TRANSREID_WEIGHT_REGISTRY = {
    'msmt17': {
        'file_id': '1x6Na97ycxS0t2Dn_0iRKWe1U5ccIqASK',
        'filename': 'vit_transreid_msmt.pth',
        'config': 'configs/MSMT17/vit_transreid_stride.yml',
        'num_classes': 1041,
        'camera_num': 15,
        'view_num': 0,
    },
    'market1501': {
        'file_id': '11p4RjmpCGGAS-876VEt7OoFrUeHTUlyO',
        'filename': 'vit_transreid_market.pth',
        'config': 'configs/Market/vit_transreid_stride.yml',
        'num_classes': 751,
        'camera_num': 6,
        'view_num': 0,
    },
    'dukemtmc': {
        'file_id': '1BipxoqyThefQviJzuJIKtFJvNblIlPGN',
        'filename': 'vit_transreid_duke.pth',
        'config': 'configs/DukeMTMC/vit_transreid_stride.yml',
        'num_classes': 702,
        'camera_num': 8,
        'view_num': 0,
    },
}


def default_assets_root():
    return Path(__file__).resolve().parent / 'external' / 'transreid'


def ensure_transreid_repo(assets_root=None, force=False):
    assets_root = Path(assets_root or default_assets_root())
    repo_dir = assets_root / 'repo'
    marker = repo_dir / 'model' / 'make_model.py'
    if marker.exists() and not force:
        patch_transreid_repo(repo_dir)
        return repo_dir

    assets_root.mkdir(parents=True, exist_ok=True)
    if repo_dir.exists():
        shutil.rmtree(str(repo_dir))

    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = Path(tmpdir) / 'transreid-main.zip'
        print('Downloading TransReID source from {}'.format(TRANSREID_REPO_URL))
        urllib.request.urlretrieve(TRANSREID_REPO_URL, str(archive_path))

        extract_dir = Path(tmpdir) / 'extract'
        with zipfile.ZipFile(str(archive_path), 'r') as zf:
            zf.extractall(str(extract_dir))

        extracted_roots = [p for p in extract_dir.iterdir() if p.is_dir()]
        if not extracted_roots:
            raise RuntimeError('Downloaded TransReID archive did not contain a source directory.')
        shutil.copytree(str(extracted_roots[0]), str(repo_dir))

    if not marker.exists():
        raise RuntimeError('TransReID source download is incomplete: {}'.format(repo_dir))
    patch_transreid_repo(repo_dir)
    return repo_dir


def patch_transreid_repo(repo_dir):
    """Apply small compatibility patches for newer PyTorch versions."""
    vit_path = Path(repo_dir) / 'model' / 'backbones' / 'vit_pytorch.py'
    if not vit_path.exists():
        return

    text = vit_path.read_text(encoding='utf-8')
    old = 'from torch._six import container_abcs'
    new = (
        'try:\n'
        '    from torch._six import container_abcs\n'
        'except ImportError:\n'
        '    import collections.abc as container_abcs'
    )
    if new in text:
        return
    if old in text:
        vit_path.write_text(text.replace(old, new), encoding='utf-8')


def download_transreid_weight(name='msmt17', assets_root=None, force=False):
    if name not in TRANSREID_WEIGHT_REGISTRY:
        raise ValueError('Unknown TransReID weight name: {}'.format(name))

    assets_root = Path(assets_root or default_assets_root())
    weight_dir = assets_root / 'weights'
    weight_dir.mkdir(parents=True, exist_ok=True)

    item = TRANSREID_WEIGHT_REGISTRY[name]
    weight_path = weight_dir / item['filename']
    if weight_path.exists() and not force:
        return weight_path

    print('Downloading TransReID {} weight to {}'.format(name, weight_path))
    try:
        import gdown
    except ImportError:
        url = 'https://drive.google.com/uc?export=download&id={}'.format(item['file_id'])
        urllib.request.urlretrieve(url, str(weight_path))
    else:
        gdown.download(id=item['file_id'], output=str(weight_path), quiet=False)

    if not weight_path.exists():
        raise RuntimeError('TransReID weight download failed: {}'.format(weight_path))
    return weight_path


def _torch_load(path, map_location='cpu'):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _import_transreid(repo_dir):
    repo_dir = Path(repo_dir).resolve()
    patch_transreid_repo(repo_dir)
    config_init = repo_dir / 'config' / '__init__.py'
    make_model_file = repo_dir / 'model' / 'make_model.py'
    if not config_init.exists() or not make_model_file.exists():
        raise FileNotFoundError(
            'TransReID source is incomplete at "{}". Run: '
            'python transreid_adapter.py --download --variant msmt17'.format(repo_dir)
        )

    repo_path = str(repo_dir)
    if repo_path in sys.path:
        sys.path.remove(repo_path)
    sys.path.insert(0, repo_path)

    # TransReID uses short top-level package names ("config", "model").
    # Drop conflicting cached modules that may come from this project or another
    # dependency before importing from the TransReID repo path.
    for module_name in ('config', 'model'):
        module = sys.modules.get(module_name)
        module_file = getattr(module, '__file__', '') if module is not None else ''
        if module is not None and not str(module_file).startswith(repo_path):
            del sys.modules[module_name]

    config_module = importlib.import_module('config')
    transreid_cfg = config_module.cfg

    try:
        model_module = importlib.import_module('model')
        make_model = getattr(model_module, 'make_model')
        if not callable(make_model):
            raise TypeError
    except (AttributeError, TypeError):
        make_model_module = importlib.import_module('model.make_model')
        make_model = make_model_module.make_model

    return transreid_cfg, make_model


def _load_config(repo_dir, config_relpath):
    transreid_cfg, _ = _import_transreid(repo_dir)
    cfg = transreid_cfg.clone()
    cfg.merge_from_file(str(Path(repo_dir) / config_relpath))
    cfg.defrost()
    cfg.MODEL.PRETRAIN_CHOICE = 'none'
    cfg.TEST.FEAT_NORM = 'yes'
    cfg.freeze()
    return cfg


def _load_checkpoint(model, weight_path, device):
    checkpoint = _torch_load(weight_path, map_location=device)
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        elif 'model' in checkpoint:
            checkpoint = checkpoint['model']

    state_dict = {}
    for key, value in checkpoint.items():
        state_dict[key.replace('module.', '')] = value

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        print('TransReID unexpected checkpoint keys: {}'.format(len(unexpected)))
    if missing:
        print('TransReID missing model keys: {}'.format(len(missing)))


def _as_pil_rgb(image):
    if isinstance(image, Image.Image):
        return image.convert('RGB')

    image = np.asarray(image)
    if image.size == 0:
        raise ValueError('empty image')

    if image.ndim == 2:
        return Image.fromarray(image.astype('uint8')).convert('RGB')

    image = image.astype('uint8')
    if image.shape[2] >= 3:
        image = image[:, :, :3][:, :, ::-1]
    return Image.fromarray(image).convert('RGB')


def extract_image_patch(image, bbox):
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


class TransReIDFeatureExtractor(object):
    """Standalone TransReID feature extractor.

    It does not change the existing project pipeline. Use this class directly, or
    use TransReIDBoxEncoder when you need the same frame+boxes interface as
    Deep SORT encoders.
    """

    def __init__(
        self,
        variant='msmt17',
        weight_path=None,
        repo_dir=None,
        assets_root=None,
        batch_size=8,
        device=None,
        camera_id=0,
        view_id=0,
        normalize=True,
        download=True,
    ):
        if variant not in TRANSREID_WEIGHT_REGISTRY:
            raise ValueError('Unknown TransReID variant: {}'.format(variant))

        self.variant = variant
        self.batch_size = batch_size
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.camera_id = camera_id
        self.view_id = view_id
        self.normalize = normalize

        if repo_dir is None:
            if not download:
                repo_dir = default_assets_root() / 'repo'
            else:
                repo_dir = ensure_transreid_repo(assets_root=assets_root)
        self.repo_dir = Path(repo_dir)

        if weight_path is None:
            if not download:
                item = TRANSREID_WEIGHT_REGISTRY[variant]
                weight_path = default_assets_root() / 'weights' / item['filename']
            else:
                weight_path = download_transreid_weight(variant, assets_root=assets_root)
        self.weight_path = Path(weight_path)

        item = TRANSREID_WEIGHT_REGISTRY[variant]
        cfg = _load_config(self.repo_dir, item['config'])
        self.cfg = cfg
        _, make_model = _import_transreid(self.repo_dir)

        self.model = make_model(
            cfg,
            num_class=item['num_classes'],
            camera_num=item['camera_num'],
            view_num=item['view_num'],
        )
        _load_checkpoint(self.model, self.weight_path, self.device)
        self.model.to(self.device)
        self.model.eval()

        self.camera_num = item['camera_num']
        self.view_num = item['view_num']
        self.feature_dim = 3840 if cfg.MODEL.JPM else 768
        if cfg.MODEL.TRANSFORMER_TYPE in ('vit_small_patch16_224_TransReID', 'deit_small_patch16_224_TransReID'):
            self.feature_dim = 1920 if cfg.MODEL.JPM else 384

        height, width = cfg.INPUT.SIZE_TEST
        self.transform = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        ])

    def _labels(self, size, camera_id=None, view_id=None):
        camera_id = self.camera_id if camera_id is None else camera_id
        view_id = self.view_id if view_id is None else view_id

        if self.camera_num > 0:
            cam = int(camera_id)
            if cam < 0 or cam >= self.camera_num:
                raise ValueError('camera_id {} is outside [0, {})'.format(cam, self.camera_num))
            cam_label = torch.full((size,), cam, dtype=torch.long, device=self.device)
        else:
            cam_label = None

        if self.view_num > 0:
            view = int(view_id)
            if view < 0 or view >= self.view_num:
                raise ValueError('view_id {} is outside [0, {})'.format(view, self.view_num))
            view_label = torch.full((size,), view, dtype=torch.long, device=self.device)
        else:
            view_label = None

        return cam_label, view_label

    def extract(self, images, camera_id=None, view_id=None):
        tensors = []
        for image in images:
            try:
                tensors.append(self.transform(_as_pil_rgb(image)))
            except ValueError:
                continue

        if not tensors:
            return torch.empty((0, self.feature_dim), dtype=torch.float32)

        features = []
        with torch.no_grad():
            for start in range(0, len(tensors), self.batch_size):
                batch = torch.stack(tensors[start:start + self.batch_size]).to(self.device)
                cam_label, view_label = self._labels(batch.size(0), camera_id, view_id)
                batch_features = self.model(batch, cam_label=cam_label, view_label=view_label)
                if isinstance(batch_features, (tuple, list)):
                    batch_features = batch_features[-1]
                if self.normalize:
                    batch_features = F.normalize(batch_features, p=2, dim=1)
                features.append(batch_features.detach().cpu())

        return torch.cat(features, 0)

    def extract_numpy(self, images, camera_id=None, view_id=None):
        return self.extract(images, camera_id=camera_id, view_id=view_id).numpy().astype(np.float32)


class TransReIDBoxEncoder(object):
    def __init__(self, *args, **kwargs):
        self.extractor = TransReIDFeatureExtractor(*args, **kwargs)
        self.feature_dim = self.extractor.feature_dim

    def __call__(self, image, boxes, camera_id=None, view_id=None):
        patches = []
        for box in boxes:
            patch = extract_image_patch(image, box)
            if patch is None:
                patch = np.zeros((256, 128, 3), dtype=np.uint8)
            patches.append(patch)
        if not patches:
            return np.empty((0, self.feature_dim), dtype=np.float32)
        return self.extractor.extract_numpy(patches, camera_id=camera_id, view_id=view_id)


def create_transreid_box_encoder(*args, **kwargs):
    return TransReIDBoxEncoder(*args, **kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description='Standalone TransReID adapter helper')
    parser.add_argument('--variant', default='msmt17', choices=sorted(TRANSREID_WEIGHT_REGISTRY.keys()))
    parser.add_argument('--assets-root', default=str(default_assets_root()))
    parser.add_argument('--download', action='store_true', help='Download TransReID source and selected weights.')
    parser.add_argument('--force', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.download:
        repo = ensure_transreid_repo(args.assets_root, force=args.force)
        weight = download_transreid_weight(args.variant, args.assets_root, force=args.force)
        print('TransReID repo: {}'.format(repo))
        print('TransReID weight: {}'.format(weight))
    else:
        print('Available variants:')
        for name, item in TRANSREID_WEIGHT_REGISTRY.items():
            print('  {} -> {}'.format(name, item['filename']))


if __name__ == '__main__':
    main()
