# Image runtime boundaries

Use this map when changing checkpoint loading, shared Qwen encoders, or the
FLUX.2 and ZImage models. The model libraries compile without the Core runtime.

## Choose the owning library

The libraries have these responsibilities and dependencies:

| Library | Responsibility | Dependencies |
| --- | --- | --- |
| `MereRunTensor` | Safetensors loading, quantized layers, sparse attention kernels, and gradient checkpointing | MLX and `MereRunModelKit` |
| `MereRunTextEncoder` | Qwen text encoder, rotary embeddings, and vision tower | MLX and `MereRunKVCache` |
| `MereRunImageModels` | FLUX.2 and ZImage transformer layers, typed configuration, and shared VAE | MLX and `MereRunTensor` |
| `MereRunCore` | Model resolution, downloads, tokenization, sampling, training, generation stages, and image output | The model libraries and runtime services |

Core re-exports the extracted libraries, so existing source imports continue
to resolve their public types. Members used only by the orchestrators use
package access. Keep model math independent of the model store, tokenizer,
CLI, and image-file decoding.

## Follow checkpoint loading

`MereRunTensor/ModelWeightsLoader.swift` chooses indexed, single-file, or
directory-shard loading. It resolves typed quantization metadata from
`MereRunModelKit`; callers supply paths, key mappings, and progress handlers.

`HFSafetensorsWeightsLoader` filters indexed shards by their declared tensor
ownership before applying mappings. Both sharded and in-memory quantized
loading use `applyQuantizedWeightsFromArrays` for module replacement. Preserve
array siblings, inferred packed dimensions, optional SVD residuals, and
nonquantized parameter updates when changing this path.

## Follow generation stages

Klein starts in `MereRunCore/Flux2Klein/Flux2KleinGenerator.swift`:

1. `+ModelLoading.swift` resolves and loads model resources.
2. `+Generation.swift` prepares the request and schedules its stages.
3. `+Encoding.swift` prepares prompt and reference-image conditioning.
4. `+Denoising.swift` runs classifier-free guidance and scheduler updates.
5. `+Output.swift` decodes the generated latents and saves the image.

`MereRunImageModels/Flux2/Flux2LatentPacking.swift` owns patch packing and
unpacking. Reference-image latents stay
fixed during denoising; only the generated-image tokens receive Euler updates.

ZImage follows the same stage split. `+Inference.swift` prepares text and
image-to-image inputs. `+Denoising.swift` owns the loop, cancellation checks,
guidance arithmetic, and tensor evaluation. `+Output.swift` owns VAE decoding,
resizing, output-directory validation, and image writing. The generator retains
cache cleanup on success and failure.

The shared Qwen model lives in `MereRunTextEncoder`. Token sampling and
autoregressive generation remain in Core; they use the encoder's tensor API.

## Validate the boundary

Compile the independent numerical test target:

```bash
swift build --target ImageRuntimeTests
```

This target depends on the model libraries and `MereRunMLXTestSupport`. The
repository gate checks transitive local and external dependencies and runs
the fixtures with the full integration suite:

```bash
./scripts/check.sh
```

The fixtures cover indexed ownership, quantized replacement and residuals,
Qwen cache and prompt-padding behavior, transformer checkpoint reloads,
batch/cache parity, and VAE scale and shift. These small generated tensors
test numerical and loading contracts. They do not establish image quality,
large-checkpoint compatibility, or GPU throughput.
