#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "coremltools==9.0",
#   "huggingface-hub==1.28.0",
#   "numpy==2.3.5",
#   "safetensors==0.8.0",
#   "torch==2.7.0",
#   "transformers==5.16.1",
# ]
# ///
"""Build Mere's pinned Core ML/MLX package for Parakeet TDT 0.6B v3.

This is audited release tooling, not an inference dependency. It downloads the
exact NVIDIA checkpoint revision, verifies the source bytes, exports the
FastConformer encoder to Core ML, maps only the compact decoder and joint to
MLX safetensors, and writes a complete file-hash closure for runtime verification.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
from typing import Any


SOURCE_REPOSITORY = "nvidia/parakeet-tdt-0.6b-v3"
SOURCE_REVISION = "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"
SOURCE_LICENSE = "CC-BY-4.0"
CONVERTER_VERSION = 4
SCHEMA_VERSION = 4
INPUT_FRAMES = 1_501
INPUT_FEATURES = 128
OUTPUT_FEATURES = 1_024
SAMPLE_RATE = 16_000
WINDOW_SECONDS = 15.0
COMPILED_MODEL_DIRECTORY = "encoder.mlmodelc"
COMPILED_DECODER_MODEL_DIRECTORY = "decoder.mlmodelc"
DECODER_EMBEDDING_FILE = "embedding.f16"
DECODER_LANES = 16
DECODER_WINDOW_FRAMES = 8
DECODER_HIDDEN_SIZE = 640
DECODER_LAYERS = 2
VOCABULARY_SIZE = 8_192
DECODER_CLASS_COUNT = 8_198
DECODER_WEIGHTS_FILE = "model.safetensors"
RUNTIME_CONFIG_FILE = "config.json"
VOCABULARY_FILE = "vocab.txt"
TOKENIZER_FILE = "tokenizer.json"
PACKAGE_FORMAT = "coreml-hybrid-v1"
DECODER_TENSOR_COUNT = 13
SOURCE_MATERIALIZATION_VERSION = 2
EXPECTED_ENVIRONMENT = {
    "python": "3.12",
    "torch": "2.7.0",
    "transformers": "5.16.1",
    "coremltools": "9.0",
    "huggingface-hub": "1.28.0",
    "numpy": "2.3.5",
    "safetensors": "0.8.0",
}
EXPECTED_XCODE = "Xcode 26.4\nBuild version 17E192"
MINIMUM_FREE_BYTES = 10 * 1_073_741_824


@dataclass(frozen=True)
class FilePin:
    byte_count: int
    sha256: str


SOURCE_FILES = {
    "config.json": FilePin(
        1_153,
        "e747b85e1bdfd300c8b8ac63bac8dd5221f8fe9bc275b48d06c735fcd6971b6e",
    ),
    "model.safetensors": FilePin(
        2_508_311_120,
        "3a2026366188c8c68598edbbff92f8d11590a08e0ae2e6775544e7b07d6a5e11",
    ),
    "processor_config.json": FilePin(
        392,
        "8346a93a3b987fa1dec57a78f045cd0817d21786589a5a096b41a57a446fd1d7",
    ),
    "tokenizer.json": FilePin(
        1_159_960,
        "bd321b096832a3f270bd3b2a88823957920f1a5c5ada71114a26ea729d0cbe91",
    ),
}

DECODER_KEY_MAP = {
    "decoder.embedding.weight": "decoder.prediction.embed.weight",
    "decoder.lstm.weight_hh_l0": "decoder.prediction.dec_rnn.lstm.0.Wh",
    "decoder.lstm.weight_ih_l0": "decoder.prediction.dec_rnn.lstm.0.Wx",
    "decoder.lstm.weight_hh_l1": "decoder.prediction.dec_rnn.lstm.1.Wh",
    "decoder.lstm.weight_ih_l1": "decoder.prediction.dec_rnn.lstm.1.Wx",
    "decoder.decoder_projector.weight": "joint.pred.weight",
    "decoder.decoder_projector.bias": "joint.pred.bias",
    "encoder_projector.weight": "joint.enc.weight",
    "encoder_projector.bias": "joint.enc.bias",
    "joint.head.weight": "joint.joint_net.2.weight",
    "joint.head.bias": "joint.joint_net.2.bias",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify_file(path: Path, pin: FilePin) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"source must be a regular non-symlink file: {path}")
    actual_bytes = path.stat().st_size
    if actual_bytes != pin.byte_count:
        raise ValueError(
            f"source byte count mismatch for {path.name}: "
            f"expected {pin.byte_count}, found {actual_bytes}"
        )
    actual_sha256 = sha256_file(path)
    if actual_sha256 != pin.sha256:
        raise ValueError(
            f"source SHA-256 mismatch for {path.name}: "
            f"expected {pin.sha256}, found {actual_sha256}"
        )


def verify_source(
    root: Path,
    source_files: dict[str, FilePin] = SOURCE_FILES,
) -> None:
    for name, pin in source_files.items():
        verify_file(root / name, pin)


def materialize_source(
    snapshot: Path,
    workspace: Path,
    source_files: dict[str, FilePin] = SOURCE_FILES,
) -> Path:
    """Copy a verified hub snapshot into a regular-file source directory."""
    destination = workspace / (
        f"source-{SOURCE_REVISION}-v{SOURCE_MATERIALIZATION_VERSION}"
    )
    if destination.exists():
        verify_source(destination, source_files)
        return destination

    repository_cache = snapshot.parent.parent.resolve()
    staging = Path(tempfile.mkdtemp(prefix=".source-", dir=workspace))
    try:
        for name, pin in source_files.items():
            snapshot_file = snapshot / name
            resolved = snapshot_file.resolve(strict=True)
            if not resolved.is_relative_to(repository_cache):
                raise ValueError(
                    f"snapshot file resolves outside its repository cache: {snapshot_file}"
                )
            verify_file(resolved, pin)
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(resolved, target)
            verify_file(target, pin)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def environment_versions() -> dict[str, str]:
    return {
        "python": ".".join(platform.python_version_tuple()[:2]),
        "torch": importlib.metadata.version("torch"),
        "transformers": importlib.metadata.version("transformers"),
        "coremltools": importlib.metadata.version("coremltools"),
        "huggingface-hub": importlib.metadata.version("huggingface-hub"),
        "numpy": importlib.metadata.version("numpy"),
        "safetensors": importlib.metadata.version("safetensors"),
    }


def validate_environment() -> str:
    actual = environment_versions()
    if actual != EXPECTED_ENVIRONMENT:
        raise ValueError(
            f"conversion environment mismatch: expected {EXPECTED_ENVIRONMENT}, found {actual}"
        )
    xcode = subprocess.run(
        ["xcodebuild", "-version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if xcode != EXPECTED_XCODE:
        raise ValueError(f"Xcode mismatch: expected {EXPECTED_XCODE!r}, found {xcode!r}")
    return xcode


def validate_destination(output: Path) -> None:
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(output.parent).free
    if free_bytes < MINIMUM_FREE_BYTES:
        raise ValueError(
            f"conversion requires at least {MINIMUM_FREE_BYTES} free bytes; found {free_bytes}"
        )


def download_source(workspace: Path) -> Path:
    from huggingface_hub import snapshot_download

    cache = workspace / "hub-cache"
    cache.mkdir(parents=True, exist_ok=True)
    snapshot = snapshot_download(
        repo_id=SOURCE_REPOSITORY,
        revision=SOURCE_REVISION,
        allow_patterns=sorted(SOURCE_FILES),
        cache_dir=cache,
    )
    return materialize_source(Path(snapshot).resolve(), workspace)


def compile_encoder(source: Path, staging: Path) -> dict[str, float]:
    import coremltools as ct
    import numpy as np
    import torch
    from transformers import ParakeetEncoder

    from parakeet_coreml_graphs import ParakeetANEEncoder

    encoder = ParakeetEncoder.from_pretrained(
        source,
        dtype=torch.float32,
        local_files_only=True,
        attn_implementation="eager",
    ).eval()
    wrapper = ParakeetANEEncoder(encoder).eval()
    example_features = torch.linspace(
        -1.0,
        1.0,
        steps=INPUT_FRAMES * INPUT_FEATURES,
        dtype=torch.float32,
    ).reshape(1, INPUT_FRAMES, INPUT_FEATURES)
    example_mask = torch.ones((1, INPUT_FRAMES), dtype=torch.float32)

    with torch.inference_mode():
        reference = encoder(
            input_features=example_features,
            attention_mask=example_mask.bool(),
            output_attention_mask=True,
        )
        eager_features, eager_mask = wrapper(example_features, example_mask)
        if not torch.equal(reference.last_hidden_state, eager_features) or not torch.equal(
            reference.attention_mask, eager_mask.int()
        ):
            raise ValueError("ANE encoder wrapper changed the pinned reference output")
        traced = torch.jit.trace(
            wrapper,
            (example_features, example_mask),
            strict=True,
        ).eval()
        traced_features, traced_mask = traced(example_features, example_mask)
    max_absolute_error = float((eager_features - traced_features).abs().max().item())
    mean_absolute_error = float((eager_features - traced_features).abs().mean().item())
    if not torch.equal(eager_mask, traced_mask) or max_absolute_error != 0.0:
        raise ValueError(
            "Torch trace parity failed: "
            f"max={max_absolute_error}, mean={mean_absolute_error}, "
            f"maskEqual={torch.equal(eager_mask, traced_mask)}"
        )

    package = staging / "ParakeetEncoder.mlpackage"
    model = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(
                name="input_features",
                shape=(1, INPUT_FRAMES, INPUT_FEATURES),
                dtype=np.float16,
            ),
            ct.TensorType(
                name="attention_mask",
                shape=(1, INPUT_FRAMES),
                dtype=np.float16,
            ),
        ],
        outputs=[
            ct.TensorType(name="encoded_features", dtype=np.float16),
            ct.TensorType(name="encoded_attention_mask", dtype=np.float16),
        ],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        skip_model_load=True,
    )
    model.author = "Mere"
    model.license = SOURCE_LICENSE
    model.short_description = "Mere FP16 Core ML encoder for NVIDIA Parakeet TDT 0.6B v3"
    model.version = str(CONVERTER_VERSION)
    model.save(package)

    subprocess.run(
        ["xcrun", "coremlcompiler", "compile", str(package), str(staging)],
        check=True,
    )
    compiled = staging / "ParakeetEncoder.mlmodelc"
    if not compiled.is_dir():
        raise ValueError(f"Core ML compiler did not create {compiled}")
    compiled.rename(staging / COMPILED_MODEL_DIRECTORY)
    shutil.rmtree(package)
    return {
        "torchTraceMaxAbsoluteError": max_absolute_error,
        "torchTraceMeanAbsoluteError": mean_absolute_error,
    }


def compile_decoder(
    source: Path,
    staging: Path,
    lanes: int = DECODER_LANES,
) -> dict[str, Any]:
    """Compile one fixed-lane TDT decoder window and its host embedding table."""
    import coremltools as ct
    import numpy as np
    import torch
    from safetensors import safe_open

    from parakeet_coreml_graphs import first_index_digits

    class DecoderWindow(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            gate_shape = (4 * DECODER_HIDDEN_SIZE, DECODER_HIDDEN_SIZE)
            self.wx0 = torch.nn.Parameter(torch.empty(gate_shape))
            self.wh0 = torch.nn.Parameter(torch.empty(gate_shape))
            self.bias0 = torch.nn.Parameter(torch.empty(4 * DECODER_HIDDEN_SIZE))
            self.wx1 = torch.nn.Parameter(torch.empty(gate_shape))
            self.wh1 = torch.nn.Parameter(torch.empty(gate_shape))
            self.bias1 = torch.nn.Parameter(torch.empty(4 * DECODER_HIDDEN_SIZE))
            self.pred = torch.nn.Linear(
                DECODER_HIDDEN_SIZE,
                DECODER_HIDDEN_SIZE,
            )
            self.enc = torch.nn.Linear(OUTPUT_FEATURES, DECODER_HIDDEN_SIZE)
            self.head = torch.nn.Linear(DECODER_HIDDEN_SIZE, DECODER_CLASS_COUNT)

        @staticmethod
        def lstm_step(
            value: torch.Tensor,
            hidden: torch.Tensor,
            cell: torch.Tensor,
            input_weight: torch.Tensor,
            hidden_weight: torch.Tensor,
            bias: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            gates = torch.nn.functional.linear(value, input_weight, bias)
            gates = gates + torch.nn.functional.linear(hidden, hidden_weight)
            input_gate, forget_gate, candidate, output_gate = gates.chunk(4, dim=-1)
            next_cell = (
                torch.sigmoid(forget_gate) * cell
                + torch.sigmoid(input_gate) * torch.tanh(candidate)
            )
            next_hidden = torch.sigmoid(output_gate) * torch.tanh(next_cell)
            return next_hidden, next_cell

        def forward(
            self,
            encoder_window: torch.Tensor,
            token_embedding: torch.Tensor,
            hidden_state: torch.Tensor,
            cell_state: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            hidden0, cell0 = self.lstm_step(
                token_embedding,
                hidden_state[0],
                cell_state[0],
                self.wx0,
                self.wh0,
                self.bias0,
            )
            hidden1, cell1 = self.lstm_step(
                hidden0,
                hidden_state[1],
                cell_state[1],
                self.wx1,
                self.wh1,
                self.bias1,
            )
            next_hidden = torch.stack((hidden0, hidden1), dim=0)
            next_cell = torch.stack((cell0, cell1), dim=0)
            pred_projection = self.pred(hidden1).unsqueeze(1)
            enc_projection = self.enc(encoder_window)
            logits = self.head(torch.relu(enc_projection + pred_projection))
            token = first_index_digits(logits[:, :, : VOCABULARY_SIZE + 1])
            duration = first_index_digits(logits[:, :, VOCABULARY_SIZE + 1 :])[..., 1]
            return (
                token.to(torch.float16),
                duration.to(torch.float16),
                next_hidden.to(torch.float16),
                next_cell.to(torch.float16),
            )

    model = DecoderWindow().eval()
    source_weights = source / "model.safetensors"
    with safe_open(source_weights, framework="pt", device="cpu") as checkpoint:
        state = model.state_dict()
        for layer in range(DECODER_LAYERS):
            state[f"wh{layer}"] = checkpoint.get_tensor(
                f"decoder.lstm.weight_hh_l{layer}"
            )
            state[f"wx{layer}"] = checkpoint.get_tensor(
                f"decoder.lstm.weight_ih_l{layer}"
            )
            state[f"bias{layer}"] = checkpoint.get_tensor(
                f"decoder.lstm.bias_hh_l{layer}"
            ) + checkpoint.get_tensor(
                f"decoder.lstm.bias_ih_l{layer}"
            )
        state["pred.weight"] = checkpoint.get_tensor("decoder.decoder_projector.weight")
        state["pred.bias"] = checkpoint.get_tensor("decoder.decoder_projector.bias")
        state["enc.weight"] = checkpoint.get_tensor("encoder_projector.weight")
        state["enc.bias"] = checkpoint.get_tensor("encoder_projector.bias")
        state["head.weight"] = checkpoint.get_tensor("joint.head.weight")
        state["head.bias"] = checkpoint.get_tensor("joint.head.bias")
        model.load_state_dict(state, strict=True)

        embedding = checkpoint.get_tensor("decoder.embedding.weight")
        expected_embedding_shape = (VOCABULARY_SIZE + 1, DECODER_HIDDEN_SIZE)
        if tuple(embedding.shape) != expected_embedding_shape:
            raise ValueError(
                "decoder embedding shape mismatch: "
                f"expected {expected_embedding_shape}, found {tuple(embedding.shape)}"
            )
        embedding_path = staging / DECODER_EMBEDDING_FILE
        embedding.to(torch.float16).contiguous().numpy().tofile(embedding_path)

    example_encoder = torch.linspace(
        -1.0,
        1.0,
        steps=lanes * DECODER_WINDOW_FRAMES * OUTPUT_FEATURES,
        dtype=torch.float32,
    ).reshape(lanes, DECODER_WINDOW_FRAMES, OUTPUT_FEATURES)
    example_embedding = torch.linspace(
        -0.5,
        0.5,
        steps=lanes * DECODER_HIDDEN_SIZE,
        dtype=torch.float32,
    ).reshape(lanes, DECODER_HIDDEN_SIZE)
    example_hidden = torch.zeros(
        (DECODER_LAYERS, lanes, DECODER_HIDDEN_SIZE),
        dtype=torch.float32,
    )
    example_cell = torch.zeros_like(example_hidden)

    with torch.inference_mode():
        eager = model(
            example_encoder,
            example_embedding,
            example_hidden,
            example_cell,
        )
        traced = torch.jit.trace(
            model,
            (example_encoder, example_embedding, example_hidden, example_cell),
            strict=True,
        ).eval()
        traced_output = traced(
            example_encoder,
            example_embedding,
            example_hidden,
            example_cell,
        )
    if not torch.equal(eager[0], traced_output[0]) or not torch.equal(
        eager[1], traced_output[1]
    ):
        raise ValueError("Torch decoder trace changed token or duration decisions")
    state_max_absolute_error = max(
        float((eager[index] - traced_output[index]).abs().max().item())
        for index in (2, 3)
    )
    if state_max_absolute_error != 0.0:
        raise ValueError(
            "Torch decoder trace changed recurrent state: "
            f"max={state_max_absolute_error}"
        )

    package = staging / "ParakeetDecoder.mlpackage"
    converted = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(
                name="encoder_window",
                shape=(lanes, DECODER_WINDOW_FRAMES, OUTPUT_FEATURES),
                dtype=np.float16,
            ),
            ct.TensorType(
                name="token_embedding",
                shape=(lanes, DECODER_HIDDEN_SIZE),
                dtype=np.float16,
            ),
            ct.TensorType(
                name="hidden_state",
                shape=(DECODER_LAYERS, lanes, DECODER_HIDDEN_SIZE),
                dtype=np.float16,
            ),
            ct.TensorType(
                name="cell_state",
                shape=(DECODER_LAYERS, lanes, DECODER_HIDDEN_SIZE),
                dtype=np.float16,
            ),
        ],
        outputs=[
            ct.TensorType(name="token", dtype=np.float16),
            ct.TensorType(name="duration", dtype=np.float16),
            ct.TensorType(name="next_hidden", dtype=np.float16),
            ct.TensorType(name="next_cell", dtype=np.float16),
        ],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        skip_model_load=True,
    )
    converted.author = "Mere"
    converted.license = SOURCE_LICENSE
    converted.short_description = (
        "Mere FP16 lane-batched Core ML decoder for NVIDIA Parakeet TDT 0.6B v3"
    )
    converted.version = str(CONVERTER_VERSION)
    converted.save(package)

    subprocess.run(
        ["xcrun", "coremlcompiler", "compile", str(package), str(staging)],
        check=True,
    )
    compiled = staging / "ParakeetDecoder.mlmodelc"
    if not compiled.is_dir():
        raise ValueError(f"Core ML compiler did not create {compiled}")
    compiled.rename(staging / COMPILED_DECODER_MODEL_DIRECTORY)
    shutil.rmtree(package)
    return {
        "lanes": lanes,
        "windowFrames": DECODER_WINDOW_FRAMES,
        "hiddenSize": DECODER_HIDDEN_SIZE,
        "layers": DECODER_LAYERS,
        "stateMaxAbsoluteError": state_max_absolute_error,
        "embeddingByteCount": embedding_path.stat().st_size,
        "embeddingSHA256": sha256_file(embedding_path),
    }


def ordered_vocabulary(tokenizer: dict[str, Any]) -> list[str]:
    vocabulary = tokenizer["model"]["vocab"]
    by_id: list[str | None] = [None] * len(vocabulary)
    for token, token_id in vocabulary.items():
        if not isinstance(token_id, int) or not 0 <= token_id < len(by_id):
            raise ValueError(f"invalid tokenizer id: {token_id!r}")
        if by_id[token_id] is not None:
            raise ValueError(f"duplicate tokenizer id: {token_id}")
        if "\n" in token or "\r" in token:
            raise ValueError(f"vocabulary token contains a line break: {token!r}")
        by_id[token_id] = token
    if any(token is None for token in by_id):
        raise ValueError("tokenizer ids are not contiguous")
    return [token for token in by_id if token is not None]


def build_runtime_config(source: Path) -> tuple[dict[str, Any], list[str]]:
    upstream = json.loads((source / "config.json").read_text(encoding="utf-8"))
    processor = json.loads(
        (source / "processor_config.json").read_text(encoding="utf-8")
    )
    tokenizer = json.loads((source / "tokenizer.json").read_text(encoding="utf-8"))
    vocabulary = ordered_vocabulary(tokenizer)
    encoder = upstream["encoder_config"]
    features = processor["feature_extractor"]
    durations = upstream["durations"]

    expected = {
        "architectures": ["ParakeetForTDT"],
        "blank_token_id": len(vocabulary),
        "decoder_hidden_size": 640,
        "durations": [0, 1, 2, 3, 4],
        "max_symbols_per_step": 10,
        "num_decoder_layers": 2,
        "vocab_size": len(vocabulary) + 1,
    }
    actual = {key: upstream[key] for key in expected}
    if actual != expected:
        raise ValueError(
            f"unsupported Parakeet decoder metadata: expected {expected}, found {actual}"
        )
    if encoder["hidden_size"] != OUTPUT_FEATURES:
        raise ValueError(f"unsupported encoder hidden size: {encoder['hidden_size']}")
    if features["sampling_rate"] != SAMPLE_RATE:
        raise ValueError(f"unsupported sample rate: {features['sampling_rate']}")
    if features["feature_size"] != INPUT_FEATURES:
        raise ValueError(f"unsupported feature count: {features['feature_size']}")

    runtime = {
        "mere": {
            "format": PACKAGE_FORMAT,
            "source_repository": SOURCE_REPOSITORY,
            "source_revision": SOURCE_REVISION,
        },
        "target": "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel",
        "model_defaults": {
            "enc_hidden": encoder["hidden_size"],
            "pred_hidden": upstream["decoder_hidden_size"],
            "joint_hidden": upstream["decoder_hidden_size"],
            "tdt_durations": durations,
            "num_tdt_durations": len(durations),
        },
        "preprocessor": {
            "sample_rate": features["sampling_rate"],
            "normalize": "per_feature",
            "window_size": features["win_length"] / features["sampling_rate"],
            "window_stride": features["hop_length"] / features["sampling_rate"],
            "window": "hann",
            "features": features["feature_size"],
            "n_fft": features["n_fft"],
            "dither": 0.00001,
            "pad_to": 0,
            "pad_value": features["padding_value"],
            "preemph": features["preemphasis"],
        },
        "encoder": {
            "feat_in": features["feature_size"],
            "n_layers": encoder["num_hidden_layers"],
            "d_model": encoder["hidden_size"],
            "n_heads": encoder["num_attention_heads"],
            "ff_expansion_factor": encoder["intermediate_size"]
            // encoder["hidden_size"],
            "subsampling_factor": encoder["subsampling_factor"],
            "self_attention_model": "rel_pos",
            "subsampling": "dw_striding",
            "conv_kernel_size": encoder["conv_kernel_size"],
            "subsampling_conv_channels": encoder["subsampling_conv_channels"],
            "pos_emb_max_len": encoder["max_position_embeddings"],
            "causal_downsampling": False,
            "use_bias": encoder["attention_bias"],
            "xscaling": encoder["scale_input"],
            "subsampling_conv_chunking_factor": 1,
        },
        "decoder": {
            "blank_as_pad": True,
            "vocab_size": len(vocabulary),
            "prednet": {
                "pred_hidden": upstream["decoder_hidden_size"],
                "pred_rnn_layers": upstream["num_decoder_layers"],
            },
        },
        "joint": {
            "num_classes": len(vocabulary),
            "vocabulary": vocabulary,
            "jointnet": {
                "joint_hidden": upstream["decoder_hidden_size"],
                "activation": upstream["hidden_act"],
                "encoder_hidden": encoder["hidden_size"],
                "pred_hidden": upstream["decoder_hidden_size"],
            },
            "num_extra_outputs": len(durations),
        },
        "decoding": {
            "greedy": {"max_symbols": upstream["max_symbols_per_step"]},
        },
    }
    return runtime, vocabulary


def build_decoder_bundle(source: Path, staging: Path) -> dict[str, Any]:
    from safetensors import safe_open
    from safetensors.torch import save_file

    source_weights = source / "model.safetensors"
    tensors = {}
    with safe_open(source_weights, framework="pt", device="cpu") as checkpoint:
        available = set(checkpoint.keys())
        required = set(DECODER_KEY_MAP)
        for layer in (0, 1):
            required.add(f"decoder.lstm.bias_hh_l{layer}")
            required.add(f"decoder.lstm.bias_ih_l{layer}")
        missing = sorted(required - available)
        if missing:
            raise ValueError(f"source decoder tensors are missing: {missing}")
        for source_key, runtime_key in DECODER_KEY_MAP.items():
            tensors[runtime_key] = checkpoint.get_tensor(source_key)
        for layer in (0, 1):
            tensors[f"decoder.prediction.dec_rnn.lstm.{layer}.bias"] = (
                checkpoint.get_tensor(f"decoder.lstm.bias_hh_l{layer}")
                + checkpoint.get_tensor(f"decoder.lstm.bias_ih_l{layer}")
            )

    if len(tensors) != DECODER_TENSOR_COUNT:
        raise ValueError(
            f"decoder tensor count mismatch: expected {DECODER_TENSOR_COUNT}, "
            f"found {len(tensors)}"
        )
    output_weights = staging / DECODER_WEIGHTS_FILE
    save_file(
        dict(sorted(tensors.items())),
        output_weights,
        metadata={
            "format": PACKAGE_FORMAT,
            "source_repository": SOURCE_REPOSITORY,
            "source_revision": SOURCE_REVISION,
        },
    )

    runtime_config, vocabulary = build_runtime_config(source)
    write_json(staging / RUNTIME_CONFIG_FILE, runtime_config)
    (staging / VOCABULARY_FILE).write_text(
        "\n".join(vocabulary) + "\n",
        encoding="utf-8",
    )
    shutil.copyfile(source / TOKENIZER_FILE, staging / TOKENIZER_FILE)
    return {
        "tensorCount": len(tensors),
        "weightsByteCount": output_weights.stat().st_size,
        "weightsSHA256": sha256_file(output_weights),
    }


def artifact_pins(root: Path, directory: str | None = None) -> list[dict[str, Any]]:
    model_root = root / directory if directory is not None else root
    pins: list[dict[str, Any]] = []
    for path in sorted(model_root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"compiled artifact contains a symlink: {path}")
        if not path.is_file():
            continue
        pins.append(
            {
                "filename": path.relative_to(root).as_posix(),
                "byteCount": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not pins:
        raise ValueError("compiled Core ML artifact contains no files")
    return pins


def build_manifest(
    root: Path,
    xcode: str,
    trace_receipt: dict[str, float],
    decoder_receipt: dict[str, Any],
    coreml_decoder_receipt: dict[str, Any],
) -> dict[str, Any]:
    versions = environment_versions()
    return {
        "schemaVersion": SCHEMA_VERSION,
        "source": {
            "repository": SOURCE_REPOSITORY,
            "revision": SOURCE_REVISION,
            "license": SOURCE_LICENSE,
        },
        "conversion": {
            "converter": Path(__file__).name,
            "converterVersion": CONVERTER_VERSION,
            "python": platform.python_version(),
            "torch": versions["torch"],
            "transformers": versions["transformers"],
            "coremltools": versions["coremltools"],
            "xcode": xcode.replace("\n", " / "),
        },
        "encoder": {
            "compiledModelDirectory": COMPILED_MODEL_DIRECTORY,
            "inputName": "input_features",
            "attentionMaskInputName": "attention_mask",
            "outputName": "encoded_features",
            "outputMaskName": "encoded_attention_mask",
            "inputFrames": INPUT_FRAMES,
            "inputFeatures": INPUT_FEATURES,
            "outputFeatures": OUTPUT_FEATURES,
            "sampleRate": SAMPLE_RATE,
            "windowSeconds": WINDOW_SECONDS,
        },
        "decoder": {
            "format": PACKAGE_FORMAT,
            "weightsFile": DECODER_WEIGHTS_FILE,
            "configFile": RUNTIME_CONFIG_FILE,
            "vocabularyFile": VOCABULARY_FILE,
            "tensorCount": decoder_receipt["tensorCount"],
        },
        "coreMLDecoder": {
            "compiledModelDirectory": COMPILED_DECODER_MODEL_DIRECTORY,
            "embeddingFile": DECODER_EMBEDDING_FILE,
            "encoderInputName": "encoder_window",
            "embeddingInputName": "token_embedding",
            "hiddenInputName": "hidden_state",
            "cellInputName": "cell_state",
            "decisionEncoding": "base128-float16",
            "tokenOutputName": "token",
            "durationOutputName": "duration",
            "hiddenOutputName": "next_hidden",
            "cellOutputName": "next_cell",
            "lanes": coreml_decoder_receipt["lanes"],
            "windowFrames": coreml_decoder_receipt["windowFrames"],
            "hiddenSize": coreml_decoder_receipt["hiddenSize"],
            "layers": coreml_decoder_receipt["layers"],
            "vocabularySize": VOCABULARY_SIZE,
        },
        "artifacts": artifact_pins(root),
        "verification": {
            "encoderTrace": trace_receipt,
            "decoderMapping": decoder_receipt,
            "coreMLDecoder": coreml_decoder_receipt,
        },
    }


def write_notice(path: Path) -> None:
    path.write_text(
        "# Parakeet Core ML/MLX package\n\n"
        "This artifact is an independent Mere conversion of "
        f"[{SOURCE_REPOSITORY}](https://huggingface.co/{SOURCE_REPOSITORY}/tree/{SOURCE_REVISION}) "
        f"at `{SOURCE_REVISION}`. The upstream model weights and this converted "
        "weight artifacts remain licensed under Creative Commons Attribution 4.0 "
        "International (CC BY 4.0). The Core ML encoder and decoder, compact "
        "MLX fallback layout, packaging, and Swift runtime are Mere code.\n",
        encoding="utf-8",
    )


def plan() -> dict[str, Any]:
    return {
        "sourceRepository": SOURCE_REPOSITORY,
        "sourceRevision": SOURCE_REVISION,
        "sourceLicense": SOURCE_LICENSE,
        "sourceFiles": {
            name: {"byteCount": pin.byte_count, "sha256": pin.sha256}
            for name, pin in sorted(SOURCE_FILES.items())
        },
        "environment": EXPECTED_ENVIRONMENT,
        "xcode": EXPECTED_XCODE.replace("\n", " / "),
        "encoder": {
            "inputShape": [1, INPUT_FRAMES, INPUT_FEATURES],
            "outputFeatures": OUTPUT_FEATURES,
            "precision": "fp16",
            "minimumDeploymentTarget": "iOS 18 / macOS 15",
        },
        "decoder": {
            "format": PACKAGE_FORMAT,
            "tensorCount": DECODER_TENSOR_COUNT,
            "weightsFile": DECODER_WEIGHTS_FILE,
            "coreMLModelDirectory": COMPILED_DECODER_MODEL_DIRECTORY,
            "embeddingFile": DECODER_EMBEDDING_FILE,
            "decisionEncoding": "base128-float16",
            "lanes": DECODER_LANES,
            "windowFrames": DECODER_WINDOW_FRAMES,
        },
    }


def convert(args: argparse.Namespace) -> None:
    workspace = args.workspace.resolve()
    output = args.output.resolve()
    validate_destination(output)
    workspace.mkdir(parents=True, exist_ok=True)
    xcode = validate_environment()
    source = download_source(workspace)

    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        trace_receipt = compile_encoder(source, staging)
        coreml_decoder_receipt = compile_decoder(source, staging)
        decoder_receipt = build_decoder_bundle(source, staging)
        write_notice(staging / "NOTICE.md")
        write_json(
            staging / "parakeet-coreml.json",
            build_manifest(
                staging,
                xcode,
                trace_receipt,
                decoder_receipt,
                coreml_decoder_receipt,
            ),
        )
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(output)


def convert_decoder_only(args: argparse.Namespace) -> None:
    workspace = args.workspace.resolve()
    output = args.output.resolve()
    validate_destination(output)
    workspace.mkdir(parents=True, exist_ok=True)
    validate_environment()
    source = download_source(workspace)

    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        receipt = compile_decoder(source, staging)
        write_json(staging / "decoder-receipt.json", receipt)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, help="Download/cache workspace")
    parser.add_argument("--output", type=Path, help="New artifact directory")
    parser.add_argument("--plan", action="store_true", help="Print the pinned conversion plan")
    parser.add_argument(
        "--decoder-only",
        action="store_true",
        help="Compile only the fixed-lane decoder qualification artifact",
    )
    args = parser.parse_args()
    if not args.plan and (args.workspace is None or args.output is None):
        parser.error("--workspace and --output are required unless --plan is used")
    return args


def main() -> None:
    args = parse_args()
    if args.plan:
        print(json.dumps(plan(), indent=2, sort_keys=True))
        return
    if args.decoder_only:
        convert_decoder_only(args)
        return
    convert(args)


if __name__ == "__main__":
    main()
