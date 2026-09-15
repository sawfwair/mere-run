# /// script
# requires-python = "==3.12.*"
# dependencies = ["torch==2.10.0", "diffusers==0.38.0", "transformers==5.17.0",
#                 "accelerate==1.15.0", "pillow==12.3.0", "numpy==2.5.3"]
# ///
"""Dump one reference block with real weights, without a full-model allocation."""
import argparse
import json
from pathlib import Path
import torch
from diffusers import QwenImageTransformer2DModel
from safetensors import safe_open
from safetensors.torch import load_file, save_file

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
p.add_argument("--fixture", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
a.output.parent.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(8)
config = json.loads((a.root / "transformer/config.json").read_text())
config["num_layers"] = 1
with torch.device("meta"):
    model = QwenImageTransformer2DModel.from_config(config)
model.pos_embed = type(model.pos_embed)(theta=10000, axes_dim=config["axes_dims_rope"], scale_rope=True)
index = json.loads((a.root / "transformer/diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
state = {}
for filename in set(index.values()):
    with safe_open(a.root / "transformer" / filename, framework="pt") as source:
        for key in model.state_dict():
            if index[key] == filename:
                state[key] = source.get_tensor(key).float()
model.load_state_dict(state, assign=True, strict=True)
del state
adapters = load_file(a.root / "marigold/depth/Log-stage2/trainables.safetensors")
class LoRA(torch.nn.Module):
    def __init__(self, base, down, up):
        super().__init__()
        self.base = base
        self.register_buffer("down", down.float())
        self.register_buffer("up", up.float())
    def forward(self, x):
        return self.base(x) + torch.nn.functional.linear(torch.nn.functional.linear(x.float(), self.down), self.up).to(x.dtype)
for name, module in list(model.named_modules()):
    key = f"Diffuser.{name}.lora_A.default.weight"
    if key in adapters:
        parent, _, leaf = name.rpartition(".")
        setattr(model.get_submodule(parent), leaf, LoRA(module, adapters[key], adapters[key.replace("lora_A", "lora_B")]))
del adapters
model = model.to("mps").eval()
prompt_dir = a.root / "marigold/qwen_text_embeddings"
prefix = "qwen_edit_2509_qwen_depth_realimg512"
prompt = torch.load(prompt_dir / f"{prefix}_prompt_embeds.pt", weights_only=True)[0:1].float().to("mps")
mask = torch.load(prompt_dir / f"{prefix}_prompt_mask.pt", weights_only=True)[0:1].bool().to("mps")
latents = load_file(a.fixture)["normalized"].to("mps")
b,c,h,w = latents.shape
packed = latents.reshape(b,c,h//2,2,w//2,2).permute(0,2,4,1,3,5).reshape(b,h*w//4,c*4)
out={"packed": packed, "prompt": prompt, "mask": mask}
with torch.no_grad():
    img = model.img_in(packed)
    txt = model.txt_in(model.txt_norm(prompt))
    temb = model.time_text_embed(torch.tensor([0.5],device="mps"), img)
    rope = model.pos_embed([[(1,h//2,w//2)]],max_txt_seq_len=txt.shape[1],device="mps")
    out.update(image_embedded=img, text_embedded=txt, timestep=temb, rope_image=torch.view_as_real(rope[0]), rope_text=torch.view_as_real(rope[1]))
    txt, img = model.transformer_blocks[0](hidden_states=img, encoder_hidden_states=txt, encoder_hidden_states_mask=mask,temb=temb,image_rotary_emb=rope)
    out.update(block_image=img,block_text=txt)
save_file({k:v.detach().cpu().contiguous() for k,v in out.items()},a.output)
print({k:(list(v.shape), str(v.dtype)) for k,v in out.items()})
