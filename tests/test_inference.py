"""Compatibility checks; no training, external datasets, or GPU required."""

import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIXTURES = ROOT / "tests" / "fixtures"


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.reference = np.load(FIXTURES / "upstream_inference.npz")
        cls.reference_info = json.loads(
            (FIXTURES / "upstream_inference.json").read_text(encoding="utf-8")
        )

    @classmethod
    def tearDownClass(cls):
        cls.reference.close()
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def modules(self):
        try:
            return importlib.import_module("utils"), importlib.import_module("test_image")
        except ImportError as error:
            self.fail(f"Modern inference entry point must import: {error}")

    def save_pair(self, mode="L"):
        paths = []
        for kind in ("ir", "vis"):
            path = self.directory / f"{kind}.png"
            Image.fromarray(self.reference[f"{mode}_{kind}_pixels"]).save(path)
            paths.append(path)
        return paths

    def test_entrypoint_imports(self):
        self.modules()

    def test_scientific_sources_remain_unchanged(self):
        for relative, expected in self.reference_info["source_sha256"].items():
            with self.subTest(file=relative):
                data = (ROOT / relative).read_bytes().replace(b"\r\n", b"\n")
                self.assertEqual(hashlib.sha256(data).hexdigest(), expected)

    def test_preprocessing_retains_reference_pixels_and_channel_order(self):
        utils, _ = self.modules()
        for mode in ("L", "RGB"):
            ir, _ = self.save_pair(mode)
            with self.subTest(mode=mode):
                result = utils.get_test_images(str(ir), mode=mode)
                self.assertEqual(result.dtype, torch.float32)
                np.testing.assert_array_equal(
                    result.numpy(), self.reference[f"{mode}_ir_tensor"]
                )

    def test_gray_resize_uses_nearest_without_contrast_stretch(self):
        utils, _ = self.modules()
        pixels = np.array([[20, 50, 90, 110], [130, 150, 170, 200]], dtype=np.uint8)
        path = self.directory / "gray.png"
        Image.fromarray(pixels).save(path)
        result = utils.get_image(str(path), height=4, width=8, mode="L")
        np.testing.assert_array_equal(result, pixels.repeat(2, axis=0).repeat(2, axis=1))
        batch = utils.get_train_images_auto([str(path)], height=4, width=8, mode="L")
        np.testing.assert_array_equal(batch.numpy(), result[None, None].astype(np.float32))

    def test_rgb_resize_keeps_hwc_pixels_for_legacy_training(self):
        utils, _ = self.modules()
        pixels = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
        path = self.directory / "rgb.png"
        Image.fromarray(pixels).save(path)
        result = utils.get_image(str(path), height=8, width=12, mode="RGB")
        np.testing.assert_array_equal(result, pixels.repeat(2, axis=0).repeat(2, axis=1))

    def test_uint8_png_roundtrip_does_not_rescale(self):
        utils, _ = self.modules()
        pixels = np.array([[[35], [70]], [[105], [140]]], dtype=np.uint8)
        path = self.directory / "saved.png"
        utils.save_images(path, pixels)
        with Image.open(path) as saved:
            self.assertEqual(saved.mode, "L")
            np.testing.assert_array_equal(np.array(saved), pixels[:, :, 0])

    def test_save_fusion_clamps_then_truncates_without_rounding(self):
        _, inference = self.modules()
        output = torch.tensor([[[[-2.0, 0.9, 127.9], [128.1, 255.1, 300.0]]]])
        path = self.directory / "clipped.png"
        inference.save_fusion(output, path)
        with Image.open(path) as saved:
            np.testing.assert_array_equal(
                np.array(saved), np.array([[0, 0, 127], [128, 255, 255]], dtype=np.uint8)
            )

    def test_non_finite_output_cannot_be_saved(self):
        _, inference = self.modules()
        path = self.directory / "invalid.png"
        output = torch.tensor([[[[float('nan'), float('inf')], [0.0, 1.0]]]])
        with self.assertRaisesRegex(ValueError, "non-finite"):
            inference.save_fusion(output, path)
        self.assertFalse(path.exists())

    def test_legacy_demo_interface_preserves_the_filename(self):
        _, inference = self.modules()
        model = inference.load_model(ROOT / "models" / "densefuse_gray.model", 1, 1, device="cpu")
        ir, vis = self.save_pair()
        result = inference.run_demo(
            model, ir, vis, self.directory / "legacy", 1,
            "auto", "densefuse", "addition", "1e2", "L",
        )
        self.assertEqual(result.name, "fusion_auto_1_network_densefuse_addition_1e2.png")
        self.assertTrue(result.is_file())
        with self.assertRaisesRegex(ValueError, "Only addition"):
            inference.run_demo(
                model, ir, vis, self.directory, 1,
                "auto", "densefuse", "attention_weight", "1e2", "L",
            )

    def test_both_checkpoints_match_reference_features_and_outputs(self):
        _, inference = self.modules()
        for mode, channels in (("L", 1), ("RGB", 3)):
            with self.subTest(mode=mode):
                case = self.reference_info["cases"][mode]
                weight = ROOT / "models" / case["weight"]
                self.assertEqual(hashlib.sha256(weight.read_bytes()).hexdigest(), case["weight_sha256"])
                model = inference.load_model(weight, channels, channels, device="cpu")
                self.assertFalse(model.training)
                ir, vis = self.save_pair(mode)
                result = inference.fuse_pair(model, ir, vis, mode=mode)
                self.assertFalse(result.requires_grad)
                self.assertTrue(torch.isfinite(result).all().item())
                tolerance = self.reference_info["tolerance"]
                np.testing.assert_allclose(result.numpy(), self.reference[f"{mode}_output"], **tolerance)
                with torch.no_grad():
                    left = model.encoder(torch.from_numpy(self.reference[f"{mode}_ir_tensor"]))
                    right = model.encoder(torch.from_numpy(self.reference[f"{mode}_vis_tensor"]))
                    fused = model.fusion(left, right)
                for actual, suffix in ((left[0], "encoded_ir"), (right[0], "encoded_vis"), (fused[0], "fused")):
                    np.testing.assert_allclose(actual.numpy(), self.reference[f"{mode}_{suffix}"], **tolerance)

    def test_wrong_checkpoint_channels_are_rejected_strictly(self):
        _, inference = self.modules()
        with self.assertRaises(RuntimeError):
            inference.load_model(ROOT / "models" / "densefuse_gray.model", 3, 3, device="cpu")

    def test_mismatched_pair_is_rejected_without_resize(self):
        _, inference = self.modules()
        model = inference.load_model(ROOT / "models" / "densefuse_gray.model", 1, 1, device="cpu")
        ir, vis = self.save_pair()
        Image.fromarray(np.zeros((16, 32), dtype=np.uint8)).save(vis)
        with self.assertRaisesRegex(ValueError, "same.*(shape|size)"):
            inference.fuse_pair(model, ir, vis, mode="L")

    def test_reflection_padding_minimum_size_is_checked(self):
        _, inference = self.modules()
        model = inference.load_model(ROOT / "models" / "densefuse_gray.model", 1, 1, device="cpu")
        ir, vis = self.save_pair()
        for path in (ir, vis):
            Image.fromarray(np.zeros((1, 8), dtype=np.uint8)).save(path)
        with self.assertRaisesRegex(ValueError, "at least 2"):
            inference.fuse_pair(model, ir, vis, mode="L")

    @unittest.skipIf(torch.cuda.is_available(), "This assertion requires a CPU-only host")
    def test_explicit_cuda_request_does_not_silently_fall_back(self):
        _, inference = self.modules()
        with self.assertRaisesRegex(ValueError, "CUDA.*(available|unavailable)"):
            inference.load_model(ROOT / "models" / "densefuse_gray.model", 1, 1, device="cuda")

    def run_cli(self, ir, vis, output, *extra):
        return subprocess.run(
            [sys.executable, "-B", str(ROOT / "test_image.py"),
             "--ir", str(ir), "--vis", str(vis), "--output", str(output),
             "--device", "cpu", *extra],
            cwd=self.directory, capture_output=True, text=True, timeout=120,
        )

    def test_cli_from_another_directory_saves_png_and_honest_metadata(self):
        ir, vis = self.save_pair()
        output = self.directory / "nested" / "fused.png"
        process = self.run_cli(ir, vis, output)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        with Image.open(output) as image:
            self.assertEqual(image.mode, "L")
            expected = self.reference["L_output"][0, 0].clip(0, 255).astype(np.uint8)
            np.testing.assert_allclose(np.array(image).astype(np.int16), expected.astype(np.int16), atol=1, rtol=0)
        record = json.loads(output.with_suffix(".png.json").read_text(encoding="utf-8"))
        self.assertEqual(record["fusion"], "arithmetic_mean")
        self.assertEqual(record["device"], "cpu")
        self.assertEqual(record["dtype"], "float32")
        self.assertIsNone(record["checkpoint_training_config"])
        self.assertEqual(record["inputs"]["ir"]["sha256"], hashlib.sha256(ir.read_bytes()).hexdigest())
        self.assertEqual(record["output"]["sha256"], hashlib.sha256(output.read_bytes()).hexdigest())
        self.assertTrue(record["raw_output"]["finite"])
        self.assertIn("source_sha256", record)
        self.assertIn("cudnn_allow_tf32", record["backend"])

    def test_cli_refuses_to_overwrite_input(self):
        ir, vis = self.save_pair()
        original = ir.read_bytes()
        result = self.run_cli(ir, vis, ir)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("overwrite", result.stderr.lower())
        self.assertEqual(ir.read_bytes(), original)

    def test_missing_images_have_an_actionable_error(self):
        result = self.run_cli(self.directory / "missing.png", self.directory / "missing2.png", self.directory / "fused.png")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not found", result.stderr.lower())


if __name__ == "__main__":
    unittest.main()
