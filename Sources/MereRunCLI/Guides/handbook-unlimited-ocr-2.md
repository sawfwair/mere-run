# Unlimited-OCR-2 (Baidu)

Use `vision-ocr-unlimited-2` for native document OCR with page images in their
original order. The model runs in Swift and MLX with the original BF16 weights.

## Example to adapt

```bash
mere.run model pull vision-ocr-unlimited-2
mere.run vision ocr page-01.png page-02.png \
  --backend unlimited --temperature 0 --max-tokens 16384
```

All images form one document. Output preserves `<PAGE>` markers. With
`--output-dir out`, multiple pages produce `out/document.txt`; a single page
uses the image stem plus `.txt`. A token-limit warning means the document can
be incomplete. Supply raster images; rasterize PDFs before invoking OCR.

## Controls and variants

`--unlimited-model` accepts `vision-ocr-unlimited-2` or a directory containing
the original checkpoint shards, index, configuration, and tokenizer files.
Quantized exports aren't supported. Pull the model explicitly before use.

`--unlimited-crop auto` uses 640px tiles for one page and one 1024px global
view per page for multiple pages. Choose `base` to disable tiles or `tiles`
to tile every page. `--compare` isn't supported for this backend.

The shared CLI defaults are 4,096 generated tokens and temperature 0.2.
Use `--temperature 0` for upstream greedy decoding. Increase `--max-tokens`
for long documents, keeping prompt plus output within 131,072 tokens.
The per-page guard forces a transition after 20,000 output tokens.

`--unlimited-no-repeat-ngram-size 35` follows the upstream serving recipe to
prevent repeated dotted lines and other token loops. Zero disables the
constraint. `--unlimited-ngram-window` defaults to 128 generated tokens for
one page or 1,024 for multiple pages. Layout tags and HTML tables are preserved.

## Sources and validation

The original checkpoint is
[baidu/Unlimited-OCR-2](https://huggingface.co/baidu/Unlimited-OCR-2), pinned to
`c14071b549f4fcab10909f3c4acc6a2edb366947`. Its tensor shapes, page positions,
SAM/CLIP encoder, DeepSeek MoE, and learned sparse DASH attention define this
runtime. The upstream model card declares MIT licensing; inherited DeepSeek
components retain their Apache-2.0 attribution.

Small scalar fixtures validate decoder logits, cached decoding, and sparse
attention equations. Full-checkpoint Metal tests on an M5 Max with 128 GiB
recover fixed anchors from an invoice, base and tiled W-9 forms, and a two-page
W-9. A receipt spelling error also occurs in the pinned upstream implementation.
The reproducible report and numerical corrections are in
`docs/qualification/unlimited-ocr-2-metal-20261030/parity/`.
JPEG decoding uses statically linked libjpeg-turbo 3.2.0 with Pillow-compatible
full-resolution 8-bit samples. It requires no Python, ignores ICC transforms
and EXIF orientation, and rejects truncated JPEGs. Other formats retain their
platform decoder. Exact receipt pixel/tensor checks and repeated OCR comparisons
are in `docs/qualification/unlimited-ocr-2-metal-20261030/jpeg/`.
The `execution/` follow-up fixes differing CLI/XCTest Metal artifacts. Both
paths agree with verified kernels. The earlier CLI `9.000` result came from
a binary that didn't match its provenance stamp and isn't a valid parity result.
The `price/` follow-up corrects DASH scalar promotion and traces the remaining
price difference through matched token histories and hidden states. Exact
transcription parity remains unproven.
The `layout/` follow-up traces the first coordinate split using upstream state
substitutions and identical-input projection probes. The remaining backend
accumulation differences don't establish another native equation or dtype fix.
These checks don't establish exact transcription, BF16 numerical parity,
sustained throughput, or long-document memory behavior. The runtime retains
all document KV rows and router keys, so memory grows with document length.
It doesn't implement the upstream SGLang streaming page-window scheduler.
