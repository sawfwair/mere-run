# Clef full-checkpoint checks — October 1, 2026

The native CLI completed text, image, video, and resized-image decisions using
the full pinned `mlx-community/clef-4bit` checkpoint on Apple Silicon.
All choice answers agreed with the pinned reference.
Text, solid-image, and video probabilities differed by at most `0.0009`.
After fixing RGB resizing, normalization, and learned-position arithmetic,
the resized-image probability difference fell from `0.0145` to `0.0031`.
Independent fixtures now match the tested media preprocessing and BF16 position
interpolation exactly; full-model numerical parity remains approximate.

[Machine-readable receipt](./receipts/clef-native-2026-10-01.json) records the
requests, media construction, results, timings, package versions, source pin,
tested binary hash, and memory measurements. These four short synthetic probes
establish execution and bounded numerical agreement, not general accuracy or
maximum-context capacity.

## Runtime and memory

The native debug CLI loaded the original affine 4-bit/group-64 backbone, BF16
vision tower, and BF16 joint head. No Python process participates in native
inference. The checkpoint occupies approximately 16.3 GB.

| Probe | Input tokens | Native complete command | Peak process footprint | Maximum probability difference |
| --- | ---: | ---: | ---: | ---: |
| Text incident routing, ordinal urgency, and truth | 339 | 7.38 s | 19.27 GB | 0.0009 |
| Solid red image, 256 × 256 | 294 | 7.85 s | 21.07 GB | 0.0001 |
| Four solid red video frames, 64 × 64 | 254 | 6.44 s | 20.92 GB | 0.0001 |
| Patterned image requiring resizing, 301 × 197 | 300 | 6.72 s | 21.28 GB | 0.0031 |

Native timing includes process startup, tokenizer and weight loading, inference,
and JSON output. Each command uses a fresh process. Peak footprint comes from
macOS `/usr/bin/time -l`; RSS alone omits part of the Metal allocation footprint.
The reference loaded once in 7.21 seconds, then evaluated the four probes in
2.62, 3.08, 3.03, and 1.98 seconds respectively. Those inference-only timings
are not directly comparable to the complete native commands.

All completed native runs used the normal admission check.

## Reference comparison

Source and checkpoint revision:
`e0a23bd4406c15075b7473616429c46f3fd130a9`.
The reference is the checkpoint's `clef_mlx.py`, with SHA-256
`f1abbe542ee3e98764d71fd5fc8489e794db5619abe50c32160ec863e1e0e7a2`.
The Python environment uses MLX and `mlx-vlm`; exact package versions are in
the receipt. No second checkpoint download is needed.

`mlx-vlm` 0.7.4 ignores the nested `video_processor` configuration in this
checkpoint's `processor_config.json` and substitutes its defaults. For this
comparison, its `Qwen3VLVideoProcessor` was explicitly constructed with the
checkpoint's declared settings, including minimum pixels 4096 and maximum
pixels 25165824. The pinned loader and neural model code were unchanged.
Without that adjustment, the small video produces 318 tokens instead of the
checkpoint-configured 254. All four native preflight token counts and field
spans matched the configured reference.

The text probe selected `technical` with native confidence `0.9765`, reported
urgency `2.9549` on a zero-based 0–3 scale, and returned truth probability
`0.9909` for customers being unable to place orders. The solid image and video
both selected red with confidence greater than `0.99`.

The resizing probe selected blue in both implementations. Native blue
confidence was `0.9109` versus reference `0.9140`; native truth probability for
predominantly red was `0.0893` versus reference `0.0899`. These remaining
numerical differences do not establish exact full-model parity.

The native preprocessing now uses the reference's antialiased RGB bicubic
filter, including fixed-point coefficients and intermediate byte rounding.
Normalization multiplies by the checkpoint's FP32 rescale factor before
mean/std arithmetic; division by 255 changes BF16 rounding for some byte values.
Video resize budgets account for padded temporal pairs when frame counts are
odd. Clef's learned vision-position interpolation uses the checkpoint's BF16
dtype for weights and accumulation.

`scripts/fixtures/export-clef-media-reference.py` reproduces independent
Pillow/NumPy/MLX fixtures without checkpoint weights. Native tests match resized
image and distinct video-frame RGB buffers byte-for-byte, all 256 normalized
byte values exactly, two odd-frame resize budgets, and BF16 position outputs
on three image/video grids. Other vision callers retain their existing FP32
position interpolation policy. Fixture hashes are recorded in the receipt.

## Reproduction

Use the rebuilt binary and the requests/media specified in the receipt:

```sh
.build/debug/mere.run model pull text-decide-clef-4bit
/usr/bin/time -l .build/debug/mere.run text decide \
  --model text-decide-clef-4bit --input request.json --pretty
```

For an independent comparison, load the same local checkpoint with the pinned
`clef_mlx.load(checkpoint, backend="vlm")`. Apply the video processor settings
described above, convert video frame paths to RGB frame arrays, and call
`logits(request)`. Softmax the logits in FP32, then use the pinned
`systemone_answer` to decode each field. Compare input-token counts, categorical
choices, and probabilities with the receipt. Keep the native and reference
processes sequential on machines with limited memory.
