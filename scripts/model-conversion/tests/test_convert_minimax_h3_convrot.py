from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors import safe_open
from safetensors.torch import save_file


CONVERTER_PATH = Path(__file__).parents[1] / "convert_minimax_h3_convrot.py"
SPEC = importlib.util.spec_from_file_location("convert_minimax_h3_convrot", CONVERTER_PATH)
assert SPEC is not None and SPEC.loader is not None
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)


def config_tensor(group_size: int) -> torch.Tensor:
    payload = json.dumps(
        {
            "format": "int8_tensorwise",
            "convrot": True,
            "convrot_groupsize": group_size,
        }
    ).encode("utf-8")
    return torch.tensor(list(payload), dtype=torch.uint8)


class MiniMaxH3ConvRotConverterTests(unittest.TestCase):
    def test_decodes_source_convrot_group_independently_from_mlx_group(self) -> None:
        self.assertEqual(
            converter.decode_convrot_group_size(config_tensor(256), "layer.comfy_quant"),
            256,
        )
        self.assertEqual(converter.MLX_GROUP_SIZE, 64)

    def test_restored_weight_matches_rotated_activation_linear(self) -> None:
        torch.manual_seed(20260810)
        group_size = 256
        codes = torch.randint(-127, 128, (3, group_size * 2), dtype=torch.int8)
        scales = torch.tensor([[0.003], [0.007], [0.011]], dtype=torch.float32)
        rotated_weight = codes.to(torch.float32) * scales
        activation = torch.randn(4, group_size * 2)
        hadamard = converter.regular_hadamard(group_size, torch.device("cpu"))

        rotated_activation = torch.matmul(
            activation.reshape(-1, group_size),
            hadamard,
        ).reshape_as(activation)
        source_output = rotated_activation @ rotated_weight.T
        restored_weight = converter.undo_convrot(
            codes,
            scales,
            hadamard,
            group_size,
        )
        restored_output = activation @ restored_weight.T

        self.assertTrue(
            torch.allclose(source_output, restored_output, rtol=2e-5, atol=2e-5),
            msg=f"maximum error: {(source_output - restored_output).abs().max().item()}",
        )

    def test_conversion_uses_each_tensor_source_group_and_mlx_group64_output(self) -> None:
        torch.manual_seed(7)
        tensors: dict[str, torch.Tensor] = {"model.bias": torch.randn(4)}
        for name, source_group in (("adaln", 64), ("attention", 256)):
            rows = 4
            tensors[f"{name}.weight"] = torch.randint(
                -127,
                128,
                (rows, source_group),
                dtype=torch.int8,
            )
            tensors[f"{name}.weight_scale"] = torch.rand(rows, 1) / math.sqrt(source_group)
            tensors[f"{name}.comfy_quant"] = config_tensor(source_group)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.safetensors"
            output = root / "output.safetensors"
            save_file(tensors, source)

            counts = converter.convert(source, output, torch.device("cpu"))

            self.assertEqual(counts, {64: 1, 256: 1})
            with safe_open(output, framework="pt", device="cpu") as archive:
                self.assertEqual(tuple(archive.get_tensor("adaln.scales").shape), (4, 1))
                self.assertEqual(tuple(archive.get_tensor("attention.scales").shape), (4, 4))
                self.assertNotIn("adaln.comfy_quant", archive.keys())
                self.assertNotIn("attention.weight_scale", archive.keys())


if __name__ == "__main__":
    unittest.main()
