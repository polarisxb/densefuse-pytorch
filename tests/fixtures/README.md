# Upstream source-behavior fixtures

`upstream_inference.npz` was generated before production edits from the unchanged
network at commit `4394b63e9295db1c6b7a5c3664551c90f0605f2b`.
`upstream_inference.json` records its runtime, source hashes, weight hashes,
input member hashes, crop rectangles, shapes, and tolerances.

This is an upstream-source reference on a modern CPU runtime, not an output from
PyTorch 0.4.1 or TensorFlow and not a paper evaluation result.

For each L/RGB case the archive stores:

- `*_ir_pixels`, `*_vis_pixels`: converted uint8 crops from the bundled images;
- `*_ir_tensor`, `*_vis_tensor`: the exact NCHW float32 model inputs;
- `*_encoded_ir`, `*_encoded_vis`: the two encoded feature tensors;
- `*_fused`: arithmetic-mean fused features;
- `*_output`: decoder output before clamping or quantization.

Generation used Pillow conversion before cropping. Gray input was reshaped to
NCHW and cast to float32. RGB used the original torchvision ToTensor -> float ->
NumPy -> multiply by 255 -> stack -> float32 sequence. The original encoder,
fusion, and decoder were called under no_grad with eval mode, CPU, one thread,
and the included strict-loaded weights. No old utils module was imported or
patched; preprocessing was spelled out and is covered by separate I/O tests.

To reconstruct a reference, check out the tagged source in a separate directory,
use the exact metadata runtime and inputs/crops, then run that generation sequence.
Do not overwrite these fixtures from a changed model to make a regression pass.

Source checksums normalize CRLF to LF so Git checkout line endings do not break
the scientific-source guard. Input/weight hashes refer to original file bytes.
Floating comparisons use atol=1e-4, rtol=1e-5. PNG tests allow at most one gray
level relative to quantized floating fixtures across runtimes; dedicated uint8
I/O tests require exact pixels. This tolerance is not yet a CUDA guarantee.
