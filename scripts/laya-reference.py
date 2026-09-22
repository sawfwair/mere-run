#!/usr/bin/env python3
"""Generate native Laya parity fixtures using the pinned, separately checked-out SDK.

Reference only. The Swift runtime does not execute this script or import Python.
Use torch, transformers==5.0.0, safetensors and numpy in an isolated environment.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess

import torch
from safetensors.torch import save_file
from transformers import ModernBertConfig, ModernBertModel

SDK_REVISION = "573e5b62696ba441230cd6be71d593331b5d23af"


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != SDK_REVISION:
        raise ValueError(f"Expected reference SDK {SDK_REVISION}, got {revision}")
    spec = importlib.util.spec_from_file_location("laya_common", args.upstream / "laya/common.py")
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    torch.manual_seed(173)
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = ModernBertConfig(
        hidden_size=64, intermediate_size=96, num_hidden_layers=4, num_attention_heads=2,
        vocab_size=128, pad_token_id=0, cls_token_id=1, sep_token_id=2,
        max_position_embeddings=256, local_attention=128, global_attn_every_n_layers=3,
        reference_compile=False, attn_implementation="sdpa",
        rope_parameters={
            "full_attention": {"rope_type": "default", "rope_theta": 160000.0},
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
        },
    )
    model = common.DecisionModel(ModernBertModel(cfg), head_layers=2).eval()
    cfg.save_pretrained(args.output / "encoder")
    agent = {"encoder": "synthetic-modernbert", "head_layers": 2, "max_len": 192,
             "head_max_len": 64, "act_costs": {"escalate": 0.5},
             "temperature": [1.6, 1.25, 1.98], "temperature_by_options": {"choice:11+": 0.1}}
    (args.output / "rl_agent_config.json").write_text(json.dumps(agent, indent=2) + "\n")
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, args.output / "model.safetensors")
    ids = torch.randint(5, 128, (3, 143))
    att = torch.ones_like(ids)
    att[1, 99:] = 0
    att[2, 121:] = 0
    ids[att == 0] = 0
    positions = torch.tensor([[12, 37, 72], [19, 29, 0], [21, 0, 0]])
    markers = torch.tensor([[True, True, True], [True, True, False], [True, False, False]])
    types = torch.tensor([0, 2, 1])
    with torch.no_grad():
        logits, actions = model(ids, att, positions, markers, types)
        single_logits, single_actions = model(ids[2:3], att[2:3], positions[2:3, :1], markers[2:3, :1], types[2:3])
    save_file({"input_ids": ids, "attention_mask": att, "marker_positions": positions,
               "marker_mask": markers, "question_types": types, "logits": logits,
               "action_logits": actions, "single_logits": single_logits,
               "single_action_logits": single_actions}, args.output / "reference.safetensors")
    (args.output / "provenance.json").write_text(json.dumps({
        "sdk_repository": "https://github.com/NandhaKishorM/laya", "sdk_revision": revision,
        "transformers": "5.0.0", "torch": torch.__version__, "seed": 173,
        "purpose": "Synthetic full graph, local/global attention, uneven padding, all question types and single option parity",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
