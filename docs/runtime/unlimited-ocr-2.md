# Unlimited-OCR-2

Run Baidu's [Unlimited-OCR-2](https://huggingface.co/baidu/Unlimited-OCR-2)
with native Swift and MLX on Apple Silicon. The runtime loads the original
BF16 checkpoint at revision `c14071b549f4fcab10909f3c4acc6a2edb366947`.

Install the model explicitly:

```bash
mere.run model pull vision-ocr-unlimited-2
```

Pass page images in document order. One invocation parses one document:

```bash
mere.run vision ocr page-01.png page-02.png \
  --backend unlimited --temperature 0 --max-tokens 16384
```

For a local copy of the original checkpoint, add
`--unlimited-model /path/to/Unlimited-OCR-2`. Both safetensors shards, the tensor
index, `config.json`, and tokenizer files must be present. Quantized exports
and other Unlimited-OCR versions aren't accepted.

`--unlimited-crop auto` uses dynamic 640px tiles for one page and one 1024px
global view per page for multiple pages, following upstream inference.
Use `base` to disable tiles or `tiles` to tile every page. Gray letterboxing,
row separators, and local-before-global feature order follow the reference.
JPEG inputs use a private, statically linked libjpeg-turbo 3.2.0 decoder to
match Pillow's full-resolution 8-bit RGB samples. It uses integer slow DCT and
fancy chroma upsampling, supports grayscale and CMYK, and rejects truncated
inputs. It doesn't apply ICC transforms or EXIF orientation. Other image
formats use the platform decoder. Python isn't required.
The output retains `<PAGE>` markers. `--output-dir` writes `document.txt` for
multiple pages or the image stem plus `.txt` for one page. Diagnostics go to
stderr. `--compare` isn't supported for this document backend.

The shared CLI defaults are 4,096 output tokens and temperature 0.2. Use
`--temperature 0` for upstream greedy decoding and increase `--max-tokens` for
long documents. A token-limit warning means the output can be incomplete.
The page guard prevents EOS before the expected page count and forces a page
transition after 20,000 tokens on a page. It doesn't guarantee correct OCR.

The upstream serving recipe's 35-token no-repeat constraint prevents loops on
dotted form lines. It examines generated tokens within a 128-token window for
one page or a 1,024-token window for multiple pages. Set
`--unlimited-no-repeat-ngram-size 0` to disable it for comparisons, or use
`--unlimited-ngram-window` to set the window explicitly. Output retains the
model's `<BLOCK>` and `<FORK>` layout tags, boxes, and HTML tables.

## Memory and context

The decoder combines a 128-token local window with up to twelve learned
32-token reference blocks. Routing uses head summaries, top-16 entmax-1.5
priors, page candidate constraints, and centered log-prior attention bias.
The gathered attention working set and SAM relative-position bias are chunked.

This implementation retains all document KV rows and router keys. Memory and
router summary work grow with document length. The combined prompt and output
budget must fit within 131,072 tokens. This implementation doesn't provide
bounded-memory, unlimited-length streaming or the upstream SGLang page-window
scheduler. PDF rasterization is external; supply the resulting page images.

## Validation

Small fixtures check sparse attention against an independent scalar reference,
including page selection, remote/local overlap, partial blocks, log-prior bias,
and prefill/decode agreement. Other tests check page coordinates, entmax, RoPE,
the published configuration, checkpoint shapes, and CLI routing.

The original full checkpoint also runs on Metal. On an M5 Max with 128 GiB,
single-page invoice and W-9 checks pass, including a 12-tile form, and the
two-page W-9 preserves page order and content. Both receipt crop modes miss
the same exact spelling anchor as the pinned upstream PyTorch implementation.
See the [numerical parity follow-up](../qualification/unlimited-ocr-2-metal-20261030/parity/README)
for activation, scalar, RoPE, and attention precision corrections, input-decoder
comparisons, and repeated document tests. The original qualification outputs
remain linked from that report.
The [native JPEG follow-up](../qualification/unlimited-ocr-2-metal-20261030/jpeg/README)
records exact receipt pixel/tensor parity, JPEG/PNG output comparisons, and
remaining differences between CLI, XCTest, and upstream output.
The [execution follow-up](../qualification/unlimited-ocr-2-metal-20261030/execution/README)
fixes a Metal artifact mismatch behind the CLI/XCTest difference. Both paths
agree with verified kernels; the earlier CLI `9.000` result used a binary that
didn't match its provenance stamp. Upstream transcription parity remains limited.
The [price-decision follow-up](../qualification/unlimited-ocr-2-metal-20261030/price/README)
corrects DASH scalar promotion and replays identical histories and hidden states.
The full native receipt still emits `9.00`; upstream also selects `00` when
conditioned on the corrected native layout prefix. Earlier BF16 state and layout
differences remain relevant to exact transcription.
The [first-layout-token investigation](../qualification/unlimited-ocr-2-metal-20261030/layout/README)
uses upstream activation substitutions and identical-input projection probes to
trace the first coordinate split. Backend accumulation differences remain;
the investigation doesn't establish another native equation or dtype correction.

These small document checks don't establish exact transcription, full BF16
numerical parity, sustained throughput, or long-document memory behavior.
The implementation executes no upstream Python or serving process.
