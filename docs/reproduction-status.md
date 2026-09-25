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

## Still unverified

- The selected Python 3.11 / torch 2.7.1 CUDA environment running the migrated
  DenseFuse end to end. Only its basic A10 convolution smoke test was reported
  by the user before migration.
- Historical PyTorch or original TensorFlow numerical equivalence.
- Full COCO training, resume equivalence, paper evaluation metrics, and all-pair
  benchmark results.

## Next server check

After transferring this migration branch to the server and activating the
already validated Conda environment, run inside an allocated GPU job:

```bash
python -B -m unittest discover -s tests -v
python -m zipfile -e images/IV_images.zip images
python -B test_image.py --device cuda --output outputs/a10-gray-1.png
```

Inspect both `a10-gray-1.png` and `a10-gray-1.png.json`. Record raw shape, finite
status, actual GPU/backend flags, and hashes. The CPU reference-fixture tests
remain CPU tests even when launched on a GPU host; they do not establish CUDA
equivalence. A separate CPU/CUDA numerical comparison belongs to the next
verification step. Preserve the scheduler's device allocation.
