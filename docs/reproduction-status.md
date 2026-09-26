# Reproduction status

Recorded 2026-09-26 (Asia/Shanghai). This is the first inference compatibility
milestone, not a completed paper or training reproduction.

## Verified locally

Host: Windows x86_64, Python 3.13.2, torch 2.8.0+cpu,
torchvision 0.23.0+cpu, NumPy 2.2.4, Pillow 12.3.0. Full-image runs used four
CPU threads and MKLDNN enabled; fixture tests use one CPU thread.

| Check | Result |
| --- | --- |
| Standard-library test suite | 17 tests passed |
| New inference/test code lint | Ruff default checks passed |
| Legacy utils correctness lint | Ruff E9/F63/F7/F82 passed; unrelated upstream style is not rewritten |
| Changed Python syntax | AST parsing passed |
| Frozen implementation/weights | No diff from upstream-original for net, fusion, train, args, MS-SSIM, or included weights |
| Gray/RGB weights | Strict loading, original key shapes, reference features and outputs passed |
| Gray full sample | 360×270, L, all finite |
| RGB full sample | 640×512, RGB, all finite |

Red/green evidence: before migration, the first suite had 13 failures caused by
the original import/entrypoint incompatibilities and one passing source guard.
An additional test demonstrated that direct save could conceal non-finite
pixels; save now rejects them before conversion. All 17 tests subsequently passed.

Both generated images were visually inspected for dimensions, mode, and obvious
corruption. This is not a subjective or objective comparison with paper results.

## Full-size gray source comparison

Reference: load `net.py` directly from the `upstream-original` Git object, load
the original gray state_dict, convert IR1/VIS1 through Pillow L to uint8 arrays
and float32 NCHW tensors, then call encoder -> fusion -> decoder under no_grad.
Compare the migrated `fuse_pair` output in the same modern CPU runtime.

- Shape: `[1, 1, 270, 360]`.
- Maximum absolute raw-output difference: **0.0**.
- Mean absolute raw-output difference: **0.0**.
- Decoded PNG pixels equal the reference clamp/uint8 result: **true**.
- Raw gray output range: approximately `[42.95035, 211.43246]`.
- Gray PNG SHA256: `4dc5a4a95780a298fe76cfa5b07fd858d0b472dcb703ade5a630ccc104245190`.
- Gray decoded-pixel SHA256: `49acebb5840c6f2ce36eb593337a47718cf05df0db81992f09a43eceabe85fd8`.
- RGB PNG SHA256: `3fadae7dda6b576c0eeb1c1dfcfc29ac5430d0bc02545970fd3d07b5c3ff9000`.
- RGB decoded-pixel SHA256: `e8a1be3a8d57e4198bba6ae9d5ae5967aa7a7ccd74f7026fe380418dd42851ac`.

File hashes describe these local outputs, not a cross-version PNG encoding
guarantee. Floating fixture tolerances are specified in tests/fixtures. The RGB
full image was run successfully; intermediate RGB regression uses its fixed crop.

## Review and scope

Independent read-only architecture/I/O review checked the legacy SciPy source
and the preservation boundaries. A subsequent independent final code-review
attempt was unavailable because the tool reached its usage limit. Local
self-review, numerical comparisons, lint, and automated tests were completed;
do not interpret this as an independent final code-review approval.

## Server A10 evidence

The user ran the commands on the school server and provided terminal output,
the inference JSON, and the comparison JSON. The assistant checked source hashes
against Git blobs and input/weight hashes against the local assets. The server
PNG and raw output tensors were not transferred for independent inspection.
The transcribed measurements and provenance are preserved in
[a10-gray-1.json](experiments/a10-gray-1.json); absolute home paths are omitted.

- Tested revision: `d73cbf8ff69d2173fd0ae6cadf03737e13b799d3`, clean checkout.
- Source hashes match that commit. Windows CRLF vs Linux LF explains the earlier
  differences between some raw working-file hashes; normalized content matches.
- Environment: Linux x86_64/glibc 2.28, Python 3.11.16, torch 2.7.1+cu118,
  torchvision 0.22.1+cu118, NumPy 1.26.4, Pillow 10.4.0, cuDNN 90100.
- Device: NVIDIA A10, cuda:0; driver 550.54.14 from the user's environment report.
- Regression suite: `Ran 17 tests`, `OK (skipped=1)`, 14.777 seconds. The skipped
  test requires a CPU-only host; the numerical fixture tests still run on CPU.
- Inference timestamp: `2026-09-25T17:04:00.758371+00:00`.
- IR/VIS and checkpoint SHA256 agree with the local reference assets.
- Default GPU output: `[1,1,270,360]`, all finite, raw range
  `[42.93285369873047, 211.39358520507812]`.
- Server PNG SHA256: `50cbb181fca62fbebcb9d9914f32066ac360702c8e8586e4db1f7a8e256696b9`.
- Server pixel SHA256: `a885c3f3b36776611664cfb3d558b5b8d148eb15eaee067a490754c2d1a50147`.

The following measurements use the **same server, software environment, input
pair and weights**. They do not mix the Windows/torch 2.8 reference with Linux
torch 2.7.1. CPU threads were 32. The default backend allowed cuDNN TF32, did not
allow matmul TF32, and had cuDNN benchmark/deterministic both false. The diagnostic
then disabled both TF32 flags only within its own process.

| Comparison | Max raw absolute difference | Mean raw absolute difference | Max uint8 difference | Differing pixels / 97200 |
| --- | ---: | ---: | ---: | ---: |
| CPU vs GPU default | 0.1436920166015625 | 0.016508804603859232 | 1 | 1221 (1.25617%) |
| CPU vs GPU TF32 disabled | 0.000244140625 | 0.000032724568873275947 | 1 | 1 (0.00102881%) |
| GPU default vs TF32 disabled | 0.14373779296875 | 0.01650912308398588 | 1 | 1222 (1.25720%) |
| GPU default vs its immediate repeat | 0 | 0 | 0 | 0 |

This controlled pair shows that TF32 allowance is the main source of the observed
CPU/GPU discrepancy: disabling it reduced maximum raw error about 589 times and
mean error about 504 times. A residual one-level uint8 difference is compatible
with small floating errors crossing an integer-truncation boundary; pixel-level
localization has not been supplied. Neither CPU output nor GPU output is paper
ground truth. One exact immediate repeat does not prove general determinism.

### Reference precision policy

Use explicit TF32-off runs for subsequent numerical-alignment experiments,
recording the flags and preserving existing default outputs. Do not silently
change the CLI defaults or add AMP, normalization, or output rounding to force
hash agreement. PyTorch controls convolution TF32 separately from matmul TF32;
see the [PyTorch 2.7 CUDA precision documentation](https://docs.pytorch.org/docs/2.7/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices).

For a separately named TF32-off image and normal run metadata, from the repository
root inside an allocated GPU job:

```bash
python -B - <<'PY'
import torch
from test_image import main

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
main(['--device', 'cuda', '--output', 'outputs/a10-gray-1-no-tf32.png'])
PY
```

This reuses the migrated CLI and does not edit source files. The sidecar records
the actual flags. It does not enable deterministic algorithms or promise bitwise
CPU/CUDA equality.

## All 21 gray pairs with TF32 disabled

The user next executed the supplied batch comparison command for IR1/VIS1
through IR21/VIS21. All 21 pairs completed. The diagnostic explicitly disabled
cuDNN and matmul TF32, reused one CPU and one CUDA model, and saved both PNGs
and their sidecars. The supplied functions check finite output before saving.
The transcript and exact printed aggregate are archived in
[a10-gray21-no-tf32.json](experiments/a10-gray21-no-tf32.json).

| Aggregate observation | Reported value |
| --- | ---: |
| Completed pairs | 21 / 21 |
| Total gray pixels | 6,199,556 |
| Maximum raw absolute difference | 0.000396728515625 |
| Pixel-weighted mean raw absolute difference | 0.00004171648965373742 |
| Maximum saved uint8 difference | 1 |
| Differing saved pixels | 40 |
| Differing saved pixel fraction | 0.0000064520749550451675 |
| Differing saved pixel percentage | approximately 0.0006452075% |
| Pairs with identical saved pixels | 7 |

The highest raw maximum was on pair 10. Pair 2 had the most differing saved
pixels (15); pairs 3, 5, 7, 8, 11, 13, and 17 had identical saved pixels. Counts
were independently summed from the transcript and agree with the aggregate.
Per-pair raw maxima/means were printed to eight decimal places and are preserved
as rounded strings; the missing digits are not reconstructed. The aggregate
mean is weighted by pixel count, not the unweighted mean of image means.

Server report location relative to the repository:
`outputs/a10-gray21-no-tf32-20260926T042245284922Z/summary.json`.

This is evidence of close CPU/GPU agreement across the bundled gray sample set
under the tested TF32-off procedure. The observed maxima are not a predeclared
acceptance threshold and do not establish paper quality or correctness by
themselves. No new numerical tolerance is imposed after seeing these results.
The full batch summary and per-image sidecars/PNGs have not been transferred;
their runtime, Git state and individual hashes are not independently checked.
The archive distinguishes the preceding verified environment context from the
configuration specified by the batch command.

## Still unverified

- Server RGB inference and server PNG visual inspection by the assistant.
- Historical PyTorch or original TensorFlow numerical equivalence.
- Full COCO training, resume equivalence, paper evaluation metrics, and performance
  benchmarks. All-pair gray device comparison is reported above; it is not a
  fusion-quality or speed benchmark.

## Repeating the default server check

After activating the validated Conda environment, run inside an allocated GPU job:

```bash
python -B -m unittest discover -s tests -v
python -m zipfile -e images/IV_images.zip images
python -B test_image.py --device cuda --output outputs/a10-gray-1.png
```

Inspect both `a10-gray-1.png` and `a10-gray-1.png.json`. Record raw shape, finite
status, actual GPU/backend flags, and hashes. The CPU reference-fixture tests
remain CPU tests even when launched on a GPU host; they do not establish CUDA
equivalence. Single-pair and all-pair gray device comparisons are separately
reported above; cross-process repeatability remains future work. Preserve the
scheduler's device allocation.
