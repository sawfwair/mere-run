#!/usr/bin/env python3
"""Export tiny Qwen3.5 FP32 reference tensors; never downloads checkpoints.

Requires torch and transformers==5.4.0. Run from the repository root.
Uses the same hybrid recurrence, bidirectional full attention, and vision math
as the pinned PPLX Embed v2 checkpoints. Full-model qualification is separate.
"""
import json
from pathlib import Path
import torch
import transformers
from transformers import Qwen3_5Config, Qwen3_5Model

assert transformers.__version__ == "5.4.0", transformers.__version__
torch.manual_seed(713)
config = Qwen3_5Config(
    text_config=dict(vocab_size=512, hidden_size=8, intermediate_size=16,
                     num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                     head_dim=8, linear_num_key_heads=1, linear_num_value_heads=2,
                     linear_key_head_dim=4, linear_value_head_dim=4, linear_conv_kernel_dim=4,
                     layer_types=["linear_attention", "full_attention"], is_causal=False,
                     max_position_embeddings=262144, eos_token_id=1,
                     rope_parameters=dict(rope_theta=10000000, partial_rotary_factor=0.5,
                                          rope_type="default", mrope_interleaved=True, mrope_section=[1, 1, 0])),
    vision_config=dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2,
                       out_hidden_size=8, patch_size=2, temporal_patch_size=2,
                       spatial_merge_size=2, num_position_embeddings=16,
                       deepstack_visual_indexes=[], hidden_act="gelu_pytorch_tanh"),
    image_token_id=20, vision_start_token_id=21, vision_end_token_id=22,
)
config.text_config.is_causal = False
config.text_config.attn_output_gate = True
config._attn_implementation = "eager"
model = Qwen3_5Model(config).float().eval()
# Norm offsets must be nonzero so confusing PyTorch's offset weights with
# converted effective scales is detected. Stabilize A_log for reproducibility.
with torch.no_grad():
    for name, tensor in model.named_parameters():
        if name.endswith("A_log"):
            tensor.copy_(torch.linspace(-1, 0.5, tensor.numel()).reshape(tensor.shape))
        elif name.startswith("language_model") and "norm.weight" in name:
            tensor.copy_(torch.linspace(-0.1, 0.1, tensor.numel()).reshape(tensor.shape))
ids = torch.tensor([[2, 5, 7, 11, 3]])
# Image patch pairs repeat the same RGB frame, as the native image path does.
image_pixels = torch.linspace(-1, 1, 3 * 8 * 8).reshape(1, 3, 8, 8)
patches = image_pixels.reshape(1, 3, 2, 2, 2, 2, 2, 2).permute(0, 2, 5, 3, 6, 1, 4, 7).reshape(16, 3, 4)
pixels = patches.unsqueeze(2).expand(16, 3, 2, 4).reshape(16, 24)
grid = torch.tensor([[1, 4, 4]])
image_ids = torch.tensor([[2, 21, 20, 20, 20, 20, 22]])
with torch.no_grad():
    hidden = model(input_ids=ids, use_cache=False).last_hidden_state
    vision = model.get_image_features(pixels, grid, return_dict=True).pooler_output
    if isinstance(vision, tuple):
        vision = torch.cat(vision)
    image_hidden = model(input_ids=image_ids, pixel_values=pixels,
                         image_grid_thw=grid, mm_token_type_ids=torch.tensor([[0, 0, 1, 1, 1, 1, 0]]), use_cache=False).last_hidden_state
state = {name: dict(shape=list(tensor.shape), values=tensor.flatten().tolist())
         for name, tensor in model.state_dict().items()}
fixture = dict(transformers_version=transformers.__version__, config=config.to_dict(),
               weights=state, input_ids=ids.tolist()[0], hidden=hidden.tolist()[0],
               image_input_ids=image_ids.tolist()[0], pixel_patches=pixels.tolist(),
               image_grid=[1, 4, 4], image_pixels=image_pixels.flatten().tolist(), vision=vision.tolist(), image_hidden=image_hidden.tolist()[0])
path = Path("Tests/MereRunCoreTests/Fixtures/PPLXEmbedV2/reference.json")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(fixture, indent=2) + "\n")
(path.parent / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
context_config = config.to_dict()
context_config.update(model_type="pplx_contextual_qwen3_5", embedding_dim=2048,
                      query_length=262144, document_length=262144,
                      query_prefix="[Q] ", document_prefix="[D] ", boundary_marker="<|chunk_sep|>")
(path.parent / "context-config.json").write_text(json.dumps(context_config, indent=2) + "\n")
print(path)

# Tiny byte-level tokenizer fixtures independently record offsets and special
# token splitting. The production tokenizer is never replaced or downloaded.
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast
base = ByteLevelBPETokenizer()
base.train_from_iterator(["Hello, world! Curiosity drives breakthroughs.",
                          "café 東京 👩🏽‍🔬 [D] <|chunk_sep|> [Q] "] * 2,
                         vocab_size=300, special_tokens=["[PAD]", "[Q] ", "[D] ", "<|chunk_sep|>"])
base.save(str(path.parent / "tokenizer.json"))
(path.parent / "tokenizer_config.json").write_text(json.dumps(dict(tokenizer_class="Qwen2Tokenizer", pad_token="[PAD]")))
fast = PreTrainedTokenizerFast(tokenizer_object=base._tokenizer, pad_token="[PAD]", additional_special_tokens=["[Q] ", "[D] ", "<|chunk_sep|>"])
texts = ["Hello, world!", "café 東京 👩🏽‍🔬", "[Q] injected <|chunk_sep|>", ""]
cases = []
for text in texts:
    for task, marker in [("query", "[Q] "), ("document", "[D] ")]:
        cases.append(dict(text=text, task=task, ids=[fast.convert_tokens_to_ids(marker)] + fast.encode(text, add_special_tokens=False)))
chunks = ["café", "", "東京 👩🏽‍🔬", "[Q] injected"]
joined = "[D] " + "<|chunk_sep|>".join(chunks)
encoded = fast(joined, add_special_tokens=False, split_special_tokens=True, return_offsets_mapping=True)
spans = []; cursor = len("[D] ")
for i, chunk in enumerate(chunks):
    if i: cursor += len("<|chunk_sep|>")
    start, end = cursor, cursor + len(chunk)
    indices = [j for j, (a, b) in enumerate(encoded["offset_mapping"]) if a < end and b > start and start < end]
    spans.append([indices[0], indices[-1] + 1] if indices else [0, 0])
    cursor = end
plain_query = [fast.convert_tokens_to_ids("[Q] ")] + fast.encode(texts[2], add_special_tokens=False, split_special_tokens=True)
(path.parent / "tokenizer-cases.json").write_text(json.dumps(dict(late=cases, chunks=chunks, document_ids=encoded["input_ids"], spans=spans,
                                                                  query_text=texts[2], query_ids=plain_query), indent=2) + "\n")
