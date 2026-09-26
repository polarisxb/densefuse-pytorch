# Known differences and reproduction limits

## What the compatibility patch preserves

The original network, fusion module, training script, configuration, and bundled
SSIM/MS-SSIM source are unchanged relative to `upstream-original`. Inference
retains the existing arithmetic mean, final ReLU, float32 0–255 inputs, no resize,
and clamp followed by uint8 truncation. Both supplied checkpoints load strictly.

The CLI and supported single-pair helper add input checks and device selection.
The legacy `run_demo` interface now rejects inactive strategies instead of writing
a misleading attention label. The lower-level original network still ignores its
strategy argument; its source is preserved for comparison.

## Original implementation issues intentionally deferred

| Area | Evidence | Consequence |
| --- | --- | --- |
| Fusion | `net.py`, `DenseFuse_net.fusion` | Always `(en1[0] + en2[0]) / 2`; the attention functions are unused. |
| Output activation | `net.py`, construction of `conv5` | `is_last=True` is not passed, so the output includes ReLU. |
| Loss | `train_densefuse.py`, `ssim_loss` | Uses local normalized MS-SSIM, not simply single-scale SSIM. |
| MS-SSIM product | `pytorch_msssim/__init__.py`, `torch.prod(pow1[:-1] * pow2[-1])` | Last-scale factor appears four times. Fixing this changes gradients. |
| SSIM range | `pytorch_msssim/__init__.py`, `ssim` | Range is inferred from the prediction at each scale. |
| RGB training | `utils.py`, `get_train_images_auto` | Uses HWC-to-CHW reshape rather than transpose. Do not use as a verified RGB training recipe. |
| List inference | `utils.py`, `get_test_images` | Only appends the final loop image. Supported inference passes one path per call. |
| Training devices | `train_densefuse.py` | Adam is created before CUDA migration; periodic save migrates the model and unconditionally calls CUDA afterwards. |
| Resume | `train_densefuse.py` | Loads weights only; no optimizer, iteration or RNG state. |
| Seed | `args_fusion.py` | `seed=42` is not applied. |
| Attention helper | `fusion_strategy.py` | String `is` comparisons and direct exponential normalization remain; Python may print SyntaxWarning. |
| Debug helpers | `vision_features`, `tensor_load_rgbimage`, `matSqrt` | Missing save_image_test, removed Image.ANTIALIAS, and elementwise matrix products are outside the supported path. |

These are original implementation issues, not permission to silently correct the
scientific baseline. Each future correction needs a separate experiment and
explicit comparison to the preserved version.

## Image compatibility boundary

SciPy 1.2.1 implemented the relevant image operations through Pillow. For the
current uint8 paths, migration uses the same L/RGB conversion, explicit NEAREST
resize, and direct uint8 save. See the original
[SciPy source](https://raw.githubusercontent.com/scipy/scipy/v1.2.1/scipy/misc/pilutil.py).

This does not reproduce all scipy.misc semantics: float-array bytescale, unusual
bit depths, and the old RGB channel-axis inference for arrays whose height or
width is exactly three are not promised compatible. `save_images` now explicitly
requires uint8. No resize is performed during supported inference. Different
Pillow versions/decoders still require pixel comparisons for new datasets.

## Paper and checkpoint provenance

The [paper](https://arxiv.org/html/1804.08361v9) describes addition by summation,
SSIM and an L2 pixel-distance notation; the current code uses averaging, custom
MS-SSIM, and mean MSE. Its C5 activation table and training setup also need
reconciliation with this port and the original TensorFlow implementation.

Neither included checkpoint records its training configuration. The old filename
label `1e2` is retained for the default demo name only; it is not proof of lambda,
dataset, or training history. Run metadata records training configuration as null.

The bundled gray ZIP contains 21 pairs; no verified paper-evaluation manifest is
included. The RGB ZIP has filenames ending in .jpg but PNG file contents. Do not
re-encode these assets to make extensions agree.

## Evidence levels

1. **Environment smoke test:** the user reported GPU convolution forward/backward
   success on A10, Python 3.11.16, torch 2.7.1+cu118, torchvision 0.22.1+cu118,
   CUDA runtime 11.8, cuDNN 90100, driver 550.54.14, glibc 2.28.
2. **Source-behavior regression:** fixtures from unchanged upstream source run on
   modern CPU, with exact runtime and hashes recorded in tests/fixtures.
3. **Migrated inference:** local tests/full-size runs and user-reported A10
   inference plus CPU/GPU/TF32 comparisons (including all 21 bundled gray pairs
   with TF32 disabled) are recorded in
   `docs/reproduction-status.md`. Server test output was OK with one expected skip.
4. **Historical/paper equivalence:** not established. No historical GPU output,
   original TensorFlow comparison, training run, or paper metrics are claimed.

Float32 does not itself disable TF32 or guarantee CPU/CUDA equality. The run
record captures actual backend flags; performance and precision settings are not
silently changed by the migration.

For the first gray pair on the reported A10, disabling TF32 reduced maximum
CPU/GPU raw error from 0.1436920166015625 to 0.000244140625, and differing saved
pixels from 1221 to 1 of 97200. Both comparisons had a maximum uint8 difference
of one level. This supports explicit TF32-off reference comparisons, not a
blanket claim of cross-device equality. The default GPU path repeated exactly
once in the same process; that is not a determinism guarantee across runs or GPUs.

Across all 21 bundled gray pairs, the reported TF32-off maximum raw difference
was 0.000396728515625; 40 of 6,199,556 saved pixels differed, each by one level.
This is an observed sample-set baseline, not an independently chosen error
threshold, a repeated-run determinism test, or a paper quality evaluation. The
complete batch JSON and PNGs have not been supplied for local inspection; the
archive preserves the user's terminal aggregate and rounded per-pair output.

A later user-provided contact sheet of pairs 02, 10 and 21 was visually inspected.
It shows information from both modalities with visibly attenuated contrast,
particularly on 10 and 21, and no obvious gross rendering failure. This is not a
paper-reference comparison, and the individual full-resolution server PNGs
remain unavailable. No contrast enhancement or model changes were made.
