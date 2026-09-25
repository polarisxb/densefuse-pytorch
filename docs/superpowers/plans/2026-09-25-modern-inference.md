# DenseFuse modern inference implementation plan

> Execution: inline in the existing clean `feat/modern-pytorch` checkout, with
> independent read-only architecture and final code review. The user has approved
> the first-inference milestone; preserve the original algorithm before refactoring.

**Goal:** Produce the first pretrained fused PNG and a reproducible execution
record, with minimal compatibility changes and no new algorithm dependencies.

**Architecture:** Keep scientific code and checkpoint names unchanged. Adapt
existing image utilities and inference orchestration; keep tests, environment
definitions, and experiment evidence outside the model.

**Tech Stack:** Python 3.11, torch 2.7.1/cu118, torchvision 0.22.1, NumPy 1.26.4,
Pillow 10.4.0, SciPy 1.13.1, matplotlib 3.9.2, tqdm 4.67.1,
opencv-python-headless 4.10.0.84. Standard-library unittest for tests.

## Task 1 — Freeze original behavior and write failing compatibility tests

- [x] Verify branch/tag/HEAD and a clean working tree. All point to the original
  commit; do not recreate existing refs or change remotes.
- [x] Record architecture and scope before editing production code.
- [x] Generate a compact reference fixture from the unchanged `net.py` and
  included weights using fixed image crops; record source/weight/image hashes,
  preprocessing, runtime, shapes, and numerical tolerance in fixture metadata.
- [x] Create `tests/test_inference.py` using `unittest` with temporary image files.
  Assert gray 0–255 input values, RGB channel ordering, nearest resize, uint8 PNG
  round-trip, strict CPU loading, finite reference outputs, mismatched-pair
  rejection, PNG postprocessing, and CLI behavior from a different working directory.
- [x] Run `python -B -m unittest discover -s tests -v`; confirm the existing
  removed imports / missing CPU-compatible entry behavior cause failure.

## Task 2 — Make the minimal compatibility changes

- [x] `utils.py`: remove unused load_lua import and scipy.misc image imports;
  use Pillow L/RGB conversion, NEAREST resize, and uint8 PNG save. Preserve all
  unrelated training logic and documented original bugs.
- [x] `test_image.py`: retain the callable helpers, add explicit device and
  strict weights_only loading, no_grad, input checks, and argparse for one pair.
  Default gray assets are IR1.png/VIS1.png; default output remains the legacy
  filename. Errors must explain missing assets or unavailable CUDA.
- [x] Write a companion JSON with actual strategy, dtype, shapes, input/model
  and output hashes, runtime/backend configuration, and repository revision plus
  dirty status. Never treat the legacy `1e2` label as verified weight provenance.
- [x] Re-run focused tests until green. Do not edit net/loss to satisfy tests.

## Task 3 — Environment and research documentation

- [x] Add `requirements.txt` for pinned direct scientific dependencies,
  `requirements-cu118.txt` for the official matching torch wheels, and
  `environment.yml` for Conda Python plus pip installation. A direct-dependency
  pin file is not a full transitive environment lock.
- [x] Add `.gitignore` for generated outputs, unpacked image datasets, caches,
  local environments, and training artifacts without untracking historical files.
- [x] Expand README with attribution, branch policy, setup, data extraction,
  gray/RGB CLI, tests, environment validation, current results, and known gaps.
- [x] Document legacy loss/output/fusion differences and server validation status.

## Task 4 — Verify and finalize the first milestone

- [x] Run `python -B -m unittest discover -s tests -v` and focused static analysis;
  check syntax without writing tracked bytecode caches.
- [x] Run the gray full-size sample through the CLI on local CPU. Verify the
  360×270 PNG and companion metadata; inspect the image.
- [x] Compare unchanged scientific files to `upstream-original`; inspect git diff.
- [x] Obtain independent architecture/I/O review. Final independent code-review
  retry hit a tool usage limit; perform local self-review and document the gap.
  Additional non-finite-save regression failed first and passed after the fix.
- [x] Make one local, reviewable commit using Lore trailers. Do not push or merge.
  Report that the user-validated server environment still needs the migrated
  DenseFuse end-to-end CUDA run.
