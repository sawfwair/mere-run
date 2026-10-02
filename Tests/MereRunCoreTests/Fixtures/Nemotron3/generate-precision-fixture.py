"""Regenerate the synthetic precision boundary fixture with PyTorch 2.9.1."""
from pathlib import Path
import torch

torch.save({
    "encoder.pre_encode.proj.weight": torch.tensor([[1.001, 1.003, -1.001, -1.003]], dtype=torch.float32),
    "sortformer_modules.single_hidden_to_spks.weight": torch.tensor([[0.5, 0.50390625]], dtype=torch.bfloat16),
    "sortformer_modules.activity_head.weight": torch.ones(1),
    "preprocessor.featurizer.window": torch.ones(1),
}, Path(__file__).with_name("checkpoint-precision.pt"))
