"""Single-pair inference using the original DenseFuse network and weights."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import PIL
import torch
import torchvision

from args_fusion import args
from net import DenseFuse_net
import utils


ROOT = Path(__file__).resolve().parent
DEFAULT_NAME = 'fusion_auto_1_network_densefuse_addition_1e2.png'


def resolve_device(device='auto'):
    if device == 'auto':
        device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Supported devices are auto, cpu, cuda, and cuda:<index>')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA was requested but is not available in this environment/job')
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f'CUDA device {device.index} is not available in this job')
    return device


def load_model(path, input_nc, output_nc, device='auto'):
    device = resolve_device(device)
    model = DenseFuse_net(input_nc, output_nc)
    state = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    print(f'Model {model._get_name()}: {sum(p.numel() for p in model.parameters()):,} parameters; device={device}')
    return model


@torch.no_grad()
def _generate_fusion_image(model, strategy_type, img1, img2):
    en_r = model.encoder(img1)
    en_v = model.encoder(img2)
    fused = model.fusion(en_r, en_v, strategy_type=strategy_type)
    return model.decoder(fused)[0]


def fuse_pair(model, infrared_path, visible_path, mode='L'):
    """Return raw float32 output; never resize or normalize the input pair."""
    ir_img = utils.get_test_images(str(infrared_path), height=None, width=None, mode=mode)
    vis_img = utils.get_test_images(str(visible_path), height=None, width=None, mode=mode)
    if ir_img.shape != vis_img.shape:
        raise ValueError(f'IR and VIS must have the same shape, got {tuple(ir_img.shape)} and {tuple(vis_img.shape)}')
    if min(ir_img.shape[-2:]) < 2:
        raise ValueError('Input height and width must be at least 2 for reflection padding')
    device = next(model.parameters()).device
    output = _generate_fusion_image(model, 'addition', ir_img.to(device), vis_img.to(device))
    if not torch.isfinite(output).all().item():
        raise ValueError('Model produced non-finite pixels; no output image was saved')
    return output


def save_fusion(output, path):
    """Preserve upstream clamp -> HWC -> uint8 truncation exactly."""
    if not torch.isfinite(output).all().item():
        raise ValueError('Cannot save non-finite output pixels')
    path = Path(path)
    if path.suffix.lower() != '.png':
        raise ValueError('Use a .png output to avoid lossy encoding')
    pixels = output.detach().cpu().clamp(0, 255)[0].numpy()
    pixels = pixels.transpose(1, 2, 0).astype('uint8')
    path.parent.mkdir(parents=True, exist_ok=True)
    utils.save_images(path, pixels)
    return pixels


def run_demo(model, infrared_path, visible_path, output_path_root, index,
             fusion_type, network_type, strategy_type, ssim_weight_str, mode):
    """Retain the old callable demo interface and filename convention."""
    if strategy_type != 'addition':
        raise ValueError('Only addition (arithmetic mean) is active in this baseline')
    output_path = Path(output_path_root) / (
        f'fusion_{fusion_type}_{index}_network_{network_type}_{strategy_type}_{ssim_weight_str}.png'
    )
    if output_path.resolve() in (Path(infrared_path).resolve(), Path(visible_path).resolve()):
        raise ValueError('Output must not overwrite an input image')
    output = fuse_pair(model, infrared_path, visible_path, mode=mode)
    save_fusion(output, output_path)
    print(output_path)
    return output_path


def vision_features(feature_maps, img_type):
    # Historical debug helper, not part of the supported inference path.
    # utils.save_image_test is absent upstream; see docs/known-differences.md.
    for count, features in enumerate(feature_maps, start=1):
        for index in range(features.size(1)):
            file_name = f'feature_maps_{img_type}_level_{count}_channel_{index}.png'
            feature = features[:, index, :, :].view(1, 1, features.size(2), features.size(3))
            utils.save_image_test(feature * 255, str(ROOT / 'outputs' / 'feature_maps' / file_name))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def repository_state():
    try:
        revision = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, check=True,
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ['git', 'status', '--porcelain', '--untracked-files=normal'], cwd=ROOT,
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout
        return {'revision': revision, 'dirty': bool(status.strip())}
    except (OSError, subprocess.SubprocessError):
        return {'revision': None, 'dirty': None}


def write_run_record(path, options, model_path, output, pixels):
    device = output.device
    record = {
        'schema_version': 1,
        'kind': 'inference_run_not_paper_evaluation',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'repository': repository_state(),
        'source_sha256': {
            name: sha256(ROOT / name)
            for name in ('test_image.py', 'utils.py', 'net.py', 'fusion_strategy.py', 'args_fusion.py')
        },
        'inputs': {
            name: {'path': str(image), 'sha256': sha256(image)}
            for name, image in (('ir', options.ir), ('vis', options.vis))
        },
        'checkpoint': {'path': str(model_path), 'sha256': sha256(model_path)},
        'checkpoint_training_config': None,
        'fusion': 'arithmetic_mean',
        'mode': options.mode,
        'dtype': 'float32',
        'input_range': '0-255; no normalization or resizing',
        'device': str(device),
        'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'runtime': {
            'python': platform.python_version(), 'platform': platform.platform(),
            'torch': torch.__version__, 'torchvision': torchvision.__version__,
            'numpy': np.__version__, 'pillow': PIL.__version__,
            'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
        },
        'backend': {
            'cudnn_enabled': torch.backends.cudnn.enabled,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'mkldnn_enabled': torch.backends.mkldnn.enabled,
            'cpu_threads': torch.get_num_threads(),
        },
        'raw_output': {
            'shape': list(output.shape), 'finite': bool(torch.isfinite(output).all().item()),
            'min': output.min().item(), 'max': output.max().item(),
        },
        'output': {
            'path': str(options.output), 'sha256': sha256(options.output),
            'pixel_sha256': hashlib.sha256(pixels.tobytes()).hexdigest(),
            'pixel_shape_hwc': list(pixels.shape),
            'conversion': 'clamp(0,255), CHW to HWC, uint8 truncation',
        },
    }
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ir', type=Path, help='Registered infrared image; supply with --vis')
    parser.add_argument('--vis', type=Path, help='Registered visible image; supply with --ir')
    parser.add_argument('--model', type=Path, help='Checkpoint; defaults to the included weight for --mode')
    parser.add_argument('--mode', choices=('L', 'RGB'), default='L')
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, or cuda:<index>')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs' / DEFAULT_NAME)
    options = parser.parse_args(argv)
    if (options.ir is None) != (options.vis is None):
        parser.error('Provide both --ir and --vis, or neither to use the included sample')
    if options.ir is None:
        folder, extension = ('IV_images', 'png') if options.mode == 'L' else ('test-RGB', 'jpg')
        options.ir = ROOT / 'images' / folder / f'IR1.{extension}'
        options.vis = ROOT / 'images' / folder / f'VIS1.{extension}'
    default_weight = args.model_path_gray if options.mode == 'L' else args.model_path_rgb
    model_path = (options.model or ROOT / default_weight).expanduser().resolve()
    options.ir = options.ir.expanduser().resolve()
    options.vis = options.vis.expanduser().resolve()
    options.output = options.output.expanduser().resolve()
    metadata_path = options.output.with_suffix(options.output.suffix + '.json')
    for label, path in (('IR image', options.ir), ('VIS image', options.vis), ('Checkpoint', model_path)):
        if not path.is_file():
            parser.error(f'{label} not found: {path}. Extract the bundled images ZIPs or pass explicit paths.')
    if options.output.suffix.lower() != '.png':
        parser.error('Use a .png output to avoid lossy encoding')
    if {options.output, metadata_path} & {options.ir, options.vis, model_path}:
        parser.error('Output image and metadata must not overwrite an input or checkpoint')
    channels = 1 if options.mode == 'L' else 3
    try:
        model = load_model(model_path, channels, channels, device=options.device)
        output = fuse_pair(model, options.ir, options.vis, mode=options.mode)
        pixels = save_fusion(output, options.output)
        write_run_record(metadata_path, options, model_path, output, pixels)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(f'Saved fused image: {options.output}')
    print(f'Saved run record: {metadata_path}')


if __name__ == '__main__':
    main()
