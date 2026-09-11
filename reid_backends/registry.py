from __future__ import absolute_import, division, print_function

from pathlib import Path


def create_reid_encoder(
    backend='transreid',
    reid_model='resnet50',
    reid_weights='model_data/models/model.pth',
    reid_config=None,
    batch_size=32,
    use_gpu=None,
    device=None,
    transreid_variant='msmt17',
    transreid_weights=None,
    transreid_repo=None,
    transreid_assets_root=None,
    transreid_download=True,
    custom_reid_root='custom_reid_models',
    custom_reid_name=None,
    custom_reid_dir=None,
    custom_reid_adapter=None,
):
    backend = (backend or 'transreid').lower()

    if backend == 'transreid':
        from .transreid_backend import TransReIDBoxEncoderAdapter

        return TransReIDBoxEncoderAdapter(
            variant=transreid_variant,
            weight_path=transreid_weights,
            repo_dir=transreid_repo,
            assets_root=transreid_assets_root,
            batch_size=batch_size,
            device=device,
            download=transreid_download
        )

    if backend == 'torchreid':
        from .torchreid_backend import TorchReIDBoxEncoderAdapter

        return TorchReIDBoxEncoderAdapter(
            model_name=reid_model,
            weights_path=reid_weights,
            batch_size=batch_size,
            use_gpu=use_gpu
        )

    if backend == 'custom':
        from .custom_backend import CustomReIDBoxEncoder

        if custom_reid_dir:
            model_dir = Path(custom_reid_dir)
        elif custom_reid_name:
            model_dir = Path(custom_reid_root) / custom_reid_name
        else:
            raise ValueError(
                'When --reid-backend custom is used, set --custom-reid-dir '
                'or --custom-reid-name.'
            )

        custom_weights = reid_weights
        if custom_weights == 'model_data/models/model.pth':
            custom_weights = None
        custom_model = reid_model
        if custom_model == 'resnet50':
            custom_model = None

        return CustomReIDBoxEncoder(
            model_dir=model_dir,
            adapter_path=custom_reid_adapter,
            weights_path=custom_weights,
            config_path=reid_config,
            model_name=custom_model,
            batch_size=batch_size,
            device=device
        )

    raise ValueError('Unsupported ReID backend: {}'.format(backend))
