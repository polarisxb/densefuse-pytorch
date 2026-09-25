# DenseFuse reproduction architecture

## Scope and decision

The first deliverable is one pretrained infrared/visible fusion run on modern
PyTorch. The baseline is commit `4394b63e9295db1c6b7a5c3664551c90f0605f2b`,
tagged `upstream-original`. Work belongs on `feat/modern-pytorch`; `master`
remains the stable integration branch. A later `feat/reproduction` branch should
start from a verified migration commit, not from an unrelated implementation.

Three approaches were considered:

1. Patch the existing entry points and document their boundaries (selected).
   This keeps checkpoint keys, original algorithms, and reviews easy to compare.
2. Move everything into a new Python package immediately. This introduces module
   movement and training changes before a numerical baseline exists.
3. Preserve every old dependency in a historical environment. This may help
   later historical comparisons, but does not meet the modern-server goal.

## Current boundaries

| Boundary | Owner | Contract |
| --- | --- | --- |
| Original scientific implementation | `net.py`, `fusion_strategy.py`, `pytorch_msssim/` | Keep source and parameter names unchanged in this milestone. |
| Image compatibility | `utils.py` | Preserve uint8 image decoding, 0–255 float32 tensors, and nearest training resize. No automatic normalization, registration, or inference resize. |
| Inference orchestration | `test_image.py` | Load a strict state_dict, select one device, validate a pair, call the original encoder/fusion/decoder, and save a PNG. |
| Legacy training | `train_densefuse.py`, `args_fusion.py` | Retain the current training experiment and its known limitations; no claim of training reproduction. |
| Evidence | `tests/`, output JSON sidecar | Check preprocessing and reference tensors; record inputs, weights, environment, device, and output provenance. |
| Environment | `requirements.txt`, `requirements-cu118.txt`, `environment.yml` | Pin the selected direct dependencies; keep server validation separate from local validation. |

The inference CLI accepts one IR/VIS pair, `--mode L|RGB`, a checkpoint, an
output PNG, and `--device auto|cpu|cuda[:index]`. It does not expose inactive
attention strategies or silently resize mismatched inputs. Defaults resolve
relative to the repository; explicit relative arguments resolve from the caller's
working directory. An explicit unavailable CUDA device is an error.

The existing `load_model`, `_generate_fusion_image`, and `run_demo` entry points
remain available. The first two keep their original positional arguments;
device selection is additive. `run_demo` retains the historical default filename.

## Numerical invariants

- Shared encoder weights for both inputs; reflection padding and all original
  convolution parameters, dense concatenations, and final ReLU are preserved.
- Fusion remains `(feature_ir + feature_vis) / 2`.
- Inputs are float32 on the 0–255 scale. Inference uses the original resolution.
- RGB inference retains torchvision ToTensor followed by multiplication by 255.
- Save by clamping to [0,255], converting CHW to HWC, and casting to uint8
  (truncation, not rounding). Save PNG without contrast stretching.
- Use eval and no_grad. No AMP, compile, channels-last, or performance tuning.
- Load both included models with `weights_only=True` and `strict=True`.

Golden tensors generated from the unmodified upstream network on a documented
modern CPU runtime establish a **source-behavior regression baseline**. They are
not evidence of agreement with historical PyTorch, TensorFlow, or paper results.
Small fixed crops keep regression tests inexpensive; a full sample pair is also
run end to end. Pixel equality is required for uint8 I/O fixtures. Floating
outputs use declared tolerances when compared across PyTorch versions.

## Known differences stay explicit

The original implementation averages features, enables final ReLU, trains with
MSE plus a custom normalized MS-SSIM, repeats the last MS-SSIM factor, and does
not implement exact training resume. RGB training reshapes HWC instead of
transposing it; the multi-path test helper only appends the last image. These
are recorded, not silently corrected in an inference compatibility patch.

No third-party MS-SSIM package replaces the repository implementation. No
inherited code is assigned a new license. Original author and paper attribution
must remain visible.

## Evolution gates

1. Compatibility: imports, I/O, device management, and strict checkpoint loading.
2. Pretrained reproduction: fixed input/weight hashes, raw tensor and PNG
   comparison, then independent server CUDA validation.
3. Training reproduction: dataset manifest, baseline loss/gradients, device and
   optimizer correctness, complete resume state, and original-paper comparison.
4. Evaluation reproduction: exact evaluation pairs and independently verified
   metric definitions; publish per-image and aggregate results.
5. Refactoring: only after regression evidence, extract dataset/config/training
   components into a package while preserving old checkpoint loading.
6. Optimization: measured throughput/memory improvements with recorded numerical
   differences and a switch back to the float32 reference path.
7. Research changes: separate experiments and branches, never replacement of the
   reproduction baseline.

Future dataset, training, and evaluation interfaces will be designed against
verified experiments. Empty packages, generic model registries, and plugin
systems are deliberately unnecessary for the current single-model milestone.
