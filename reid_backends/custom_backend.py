from __future__ import absolute_import, division, print_function

import hashlib
import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .base import empty_features, extract_image_patch, to_torch_features


FACTORY_NAMES = (
    'create_extractor',
    'create_reid_extractor',
    'build_extractor',
)

CLASS_NAMES = (
    'ReIDExtractor',
    'CustomReIDExtractor',
    'ReIDAdapter',
)


def _read_config(config_path):
    if not config_path:
        return {}

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError('Custom ReID config does not exist: {}'.format(config_path))

    suffix = config_path.suffix.lower()
    with config_path.open('r', encoding='utf-8') as f:
        if suffix == '.json':
            return json.load(f) or {}

        if suffix in ('.yml', '.yaml'):
            try:
                import yaml
            except ImportError:
                raise ImportError(
                    'Reading YAML custom ReID configs requires PyYAML. '
                    'Use config.json or install pyyaml.'
                )
            return yaml.safe_load(f) or {}

    raise ValueError('Unsupported custom ReID config format: {}'.format(config_path))


def _find_config(model_dir):
    for name in ('config.json', 'config.yaml', 'config.yml'):
        candidate = model_dir / name
        if candidate.exists():
            return candidate
    return None


def _import_adapter(adapter_path):
    adapter_path = Path(adapter_path).resolve()
    if not adapter_path.exists():
        raise FileNotFoundError('Custom ReID adapter does not exist: {}'.format(adapter_path))

    model_dir = str(adapter_path.parent)
    if model_dir in sys.path:
        sys.path.remove(model_dir)
    sys.path.insert(0, model_dir)

    digest = hashlib.sha1(str(adapter_path).encode('utf-8')).hexdigest()[:12]
    module_name = '_custom_reid_adapter_{}'.format(digest)
    spec = importlib.util.spec_from_file_location(module_name, str(adapter_path))
    if spec is None or spec.loader is None:
        raise ImportError('Could not import custom ReID adapter: {}'.format(adapter_path))

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _filter_kwargs(callable_obj, kwargs):
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return kwargs

    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return kwargs

    return {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }


def _call_with_supported_kwargs(callable_obj, kwargs):
    return callable_obj(**_filter_kwargs(callable_obj, kwargs))


def _build_raw_extractor(module, kwargs):
    for name in FACTORY_NAMES:
        factory = getattr(module, name, None)
        if callable(factory):
            return _call_with_supported_kwargs(factory, kwargs)

    for name in CLASS_NAMES:
        cls = getattr(module, name, None)
        if callable(cls):
            return _call_with_supported_kwargs(cls, kwargs)

    raise AttributeError(
        'Custom ReID adapter must define one of {} or one of {}.'.format(
            ', '.join(FACTORY_NAMES),
            ', '.join(CLASS_NAMES)
        )
    )


class CustomReIDFeatureExtractor(object):
    backend = 'custom'

    def __init__(
        self,
        model_dir,
        adapter_path=None,
        weights_path=None,
        config_path=None,
        model_name=None,
        batch_size=32,
        device=None,
        normalize=True
    ):
        self.model_dir = Path(model_dir).resolve()
        if not self.model_dir.exists():
            raise FileNotFoundError('Custom ReID model folder does not exist: {}'.format(self.model_dir))

        self.adapter_path = Path(adapter_path).resolve() if adapter_path else self.model_dir / 'adapter.py'
        self.config_path = Path(config_path).resolve() if config_path else _find_config(self.model_dir)
        self.config = _read_config(self.config_path)
        self.weights_path = weights_path or self.config.get('weights_path') or self.config.get('weights')
        if self.weights_path and not os.path.isabs(str(self.weights_path)):
            self.weights_path = str(self.model_dir / self.weights_path)

        self.model_name = model_name or self.config.get('model_name') or self.model_dir.name
        self._batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.normalize = bool(self.config.get('normalize', normalize))
        self.uses_camera_id = bool(self.config.get('use_camera_id', False))
        self.uses_view_id = bool(self.config.get('use_view_id', False))
        self.feature_dim = int(self.config.get('feature_dim', 0) or 0)

        module = _import_adapter(self.adapter_path)
        kwargs = {
            'model_dir': str(self.model_dir),
            'weights_path': self.weights_path,
            'config_path': str(self.config_path) if self.config_path else None,
            'config': self.config,
            'model_name': self.model_name,
            'batch_size': self.batch_size,
            'device': self.device,
            'normalize': self.normalize,
        }
        self.raw_extractor = _build_raw_extractor(module, kwargs)

        self.feature_dim = int(getattr(self.raw_extractor, 'feature_dim', self.feature_dim) or self.feature_dim)
        self.uses_camera_id = bool(getattr(self.raw_extractor, 'uses_camera_id', self.uses_camera_id))
        self.uses_view_id = bool(getattr(self.raw_extractor, 'uses_view_id', self.uses_view_id))

    @property
    def batch_size(self):
        return getattr(self.raw_extractor, 'batch_size', self._batch_size) if hasattr(self, 'raw_extractor') else self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value
        if hasattr(self, 'raw_extractor') and hasattr(self.raw_extractor, 'batch_size'):
            self.raw_extractor.batch_size = value

    def _call_extract(self, images, camera_id=None, view_id=None):
        extract_fn = getattr(self.raw_extractor, 'extract', None)
        if extract_fn is None and callable(self.raw_extractor):
            extract_fn = self.raw_extractor
        if extract_fn is None:
            raise AttributeError('Custom ReID extractor must provide extract(images, ...).')

        kwargs = {
            'images': images,
            'camera_id': camera_id,
            'view_id': view_id,
        }
        try:
            signature = inspect.signature(extract_fn)
        except (TypeError, ValueError):
            return extract_fn(images)

        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
            return extract_fn(**kwargs)

        filtered = {
            key: value
            for key, value in kwargs.items()
            if key != 'images' and key in signature.parameters
        }
        if 'images' in signature.parameters:
            filtered['images'] = images
            return extract_fn(**filtered)
        return extract_fn(images, **filtered)

    def extract(self, images, camera_id=None, view_id=None):
        if not images:
            return empty_features(self.feature_dim)

        features = to_torch_features(self._call_extract(images, camera_id=camera_id, view_id=view_id))
        if features.numel() == 0:
            return empty_features(self.feature_dim)

        if self.normalize:
            features = F.normalize(features, p=2, dim=1)
        features = features.detach().cpu().float()

        if not self.feature_dim:
            self.feature_dim = features.size(1)
        return features

    def extract_numpy(self, images, camera_id=None, view_id=None):
        return self.extract(images, camera_id=camera_id, view_id=view_id).numpy().astype(np.float32)


class CustomReIDBoxEncoder(object):
    backend = 'custom'

    def __init__(
        self,
        model_dir,
        adapter_path=None,
        weights_path=None,
        config_path=None,
        model_name=None,
        batch_size=32,
        device=None,
        normalize=True
    ):
        self.extractor = CustomReIDFeatureExtractor(
            model_dir=model_dir,
            adapter_path=adapter_path,
            weights_path=weights_path,
            config_path=config_path,
            model_name=model_name,
            batch_size=batch_size,
            device=device,
            normalize=normalize
        )
        self.feature_dim = self.extractor.feature_dim
        self.uses_camera_id = self.extractor.uses_camera_id
        self.uses_view_id = self.extractor.uses_view_id

    def __call__(self, image, boxes, camera_id=None, view_id=None):
        patches = []
        for box in boxes:
            patch = extract_image_patch(image, box)
            if patch is None:
                patch = np.zeros((256, 128, 3), dtype=np.uint8)
            patches.append(patch)

        if not patches:
            return np.empty((0, self.feature_dim), dtype=np.float32)

        features = self.extractor.extract_numpy(patches, camera_id=camera_id, view_id=view_id)
        self.feature_dim = self.extractor.feature_dim
        return features
