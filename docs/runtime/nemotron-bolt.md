# Nemotron 3.5 Bolt

Use this guide to install NVIDIA Nemotron 3.5 Bolt 74B-A9B-24GB for local
text and image chat. The runtime loads the original packed safetensors in
Swift/MLX. You don't need Python, vLLM, or an offline conversion.

```bash
mere.run model pull vision-chat-nemotron-35-bolt-24gb
mere.run text chat --model vision-chat-nemotron-35-bolt-24gb --prompt "Explain a Bloom filter."
mere.run text chat --model vision-chat-nemotron-35-bolt-24gb --image ./diagram.png --prompt "Describe this diagram."
```

The managed download is explicit. The original weights occupy 24,079,678,824
bytes, before tokenizer files, recurrent state, image encoding, attention
caches, and execution buffers. The model name doesn't guarantee that the
whole process fits in 24 GB. Shared admission checks still apply.

The runtime uses 40 Mamba-2 blocks, eight attention blocks without rotary
embeddings, and 40 heterogeneous latent MoE blocks. Each MoE block selects
nine experts. The checkpoint combines packed IQ2_XXS expert weights, packed
Q8_0 attention and latent projections, NVFP4 Mamba projections, and BF16
embeddings and vision weights. Packed weights remain in their original
formats during execution.

Text and local image inputs, streamed generation, function tools, thinking
controls, stop sequences, and log probabilities use the shared chat path.
Audio, video input, LoRA, constrained JSON, and LiLiCorr speculative decoding
are unsupported. The source model's video capability is not an execution
claim for this integration.

The upstream checkpoint is pinned to
`nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB@96a5d63027ac013c8ef37fb438f3b4ec5fbd3ba6`.
Review the [OpenMDW-1.1 terms](https://openmdw.ai/license/1-1/) for the model
materials. The repository does not bundle the checkpoint weights.

Local structural and arithmetic tests do not establish released-checkpoint
quality, throughput, peak process memory, or long-context behavior. Those
measurements require a complete installed checkpoint and separate evidence.
