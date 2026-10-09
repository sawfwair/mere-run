# PPLX Embed v2 weight packing

The two pinned 9B sources in `convert_pplx_embed_v2_mlx.py` are converted without
training. Late uses MLX affine Q8/group-64 token embeddings and Q4/group-64
transformer projections. Context uses Q8/group-64 for both: paired diagnostics
showed substantially less vector drift than its Q4 candidate. Recurrent a/b
gates, convolution, norms, vision and final embedding projections remain FP32.

Use separate Python environments: MLX CUDA 0.32.2 requires cuBLAS 12.9, while the
PyTorch 2.8.0 reference requires cuBLAS 12.8.

```bash
python -m venv /opt/pplx-convert
/opt/pplx-convert/bin/pip install 'mlx[cuda12]==0.32.2' 'huggingface-hub==1.28.0' safetensors numpy
python -m venv /opt/pplx-reference
/opt/pplx-reference/bin/pip install 'torch==2.8.0' 'accelerate==1.10.1' 'transformers==5.4.0' 'huggingface-hub==1.28.0' Pillow

for kind in late context; do
  /opt/pplx-convert/bin/python scripts/model-conversion/convert_pplx_embed_v2_mlx.py \
    --kind "$kind" --source "/workspace/source-$kind" --output "/workspace/packed-$kind"
  /opt/pplx-reference/bin/python scripts/model-conversion/qualify_pplx_embed_v2_mlx.py \
    --kind "$kind" --source "/workspace/source-$kind" --artifact "/workspace/packed-$kind"
done
```

The converter verifies source sizes and LFS SHA-256 values, records source pins
and weight reconstruction error, preserves the upstream model card and code,
and emits explicit per-module packing metadata and a complete safetensors index.
The diagnostic independently unpacks affine integers in PyTorch, checks samples
against MLX dequantization, and compares real FP32 versus packed-weight output on
a small multilingual query/document and chunk-boundary suite. Gates are minimum
vector cosine 0.95, mean cosine 0.98, finite output and complete top-1 agreement
with the FP32 reference over eight documents. These are bounded diagnostics,
not a broad retrieval benchmark or maximum-context qualification.

`publish_pplx_embed_v2_mlx.py` requires matching conversion/config/index hashes and
passing diagnostic gates. It stages in private Sawfwair repositories, verifies
every file size and SHA-256, then makes the repositories public and verifies the
immutable commits anonymously. Pass a private token file with `--token-file`;
keep it outside the repository and remove it after publication. Download and
verify the packed files first, then run native qualification and publish from
the Mac. The Hugging Face write token stays on the Mac. Publication requires
`--apple-evidence /path/to/native-proof.json` in addition to `--token-file`.

On Apple Silicon, download the immutable artifact and run:

```bash
python3 scripts/reference-parity/check_pplx_embed_v2_native.py \
  --artifact /path/to/downloaded-model --output /path/to/native-proof.json
```

This uses the public native CLI, checks token/vector counts and cross-framework
parity, and captures macOS process memory/timing. Contextual int8 output remains
independent of model weight quantization. Image retrieval quality, cross-model
small/large alignment, and maximum-context memory behavior need separate checks.
