# Clef structured decisions

`ClefDecisionOperation` owns local tokenizer loading, checkpoint loading, media
preparation, schema token spans, and SystemOne-style answer decoding. The
`ClefJointHead` and its evidence/decoder layers live in `MereRunQwenModel`, next
to the Qwen3.5 backbone and vision tower they consume. Inference uses Swift/MLX;
Python is used only by the independent fixture exporter. Both the 27B
`text-decide-clef-4bit` and 9B `text-decide-clef-flash-4bit` checkpoints use this
operation; backbone and head dimensions come from their pinned configurations.
Flash configuration provenance lives in `Fixtures/ClefFlash`.

The request uses a questions object whose source order is retained by a typed
JSON parser. Choice option IDs sort lexically; score levels retain array order;
noul options are true then false. Structured state and option semantics use
sorted, compact JSON. Only state is truncated to the token budget: complete
schema spans and the final assistant marker remain intact.

The operation is serial and owns its loaded modules until `unload()`. Each
prediction starts with fresh backbone state and retains every final hidden
state. The lexical prior gathers rows from the output embedding matrix and
dequantizes only those rows. No autoregressive generation or vocabulary logits
are allocated. The joint head is loaded strictly in BF16 and remains unquantized.

Images are local paths. Videos are arrays of local frame paths representing a
24-fps source, uniformly sampled at 2 fps with paired temporal patches. Processor
normalization, geometry, timestamps, placeholders, and multimodal rotary
positions follow the pinned reference. Mixed image/video records are rejected.
Media stays intact; requests that exceed the context budget fail before weights
load. Preflight decodes and resizes media but does not load weights or run MLX.

Run `swift test --filter Clef` and the repository gate. Fixture provenance lives
under `Tests/MereRunCoreTests/Fixtures/Clef`; reproduce it with
`scripts/fixtures/export-clef-reference.py` and
`scripts/fixtures/export-clef-media-reference.py`. Independent fixtures cover
the tiny FP32 head, schema encoding, Pillow RGB bicubic pixels, exact FP32
normalization, odd-frame video geometry, and BF16 learned-position interpolation.
The optional real-tokenizer test compares every ID
with the pinned reference; set `MERERUN_CLEF_TOKENIZER` to a directory containing
the checkpoint's `tokenizer.json` and `tokenizer_config.json` to run it.

Full-checkpoint text, image, video, and resized-image probes complete with matching
choice answers and probability differences within 0.0031 of the configured
reference. These probes do not establish general accuracy or exact full-model
parity. See
`docs/benchmarks/clef-native-qualification-2026-10-01.md` and its receipt for the
bounded results and remaining numerical difference.

Flash full-checkpoint parity is covered by the opt-in
`ClefFlashCheckpointParityTests`. Generate independent inputs with
`scripts/fixtures/export-clef-flash-parity.py`, then set
`MERERUN_TEST_MLX_DEVICE=gpu`, `MERERUN_TEST_CLEF_FLASH_ROOT`, and
`MERERUN_TEST_CLEF_PARITY_DIR`. This test calls the runtime directly and writes
`native-runtime.json` in the parity directory; it does not exercise CLI admission.
See `docs/benchmarks/clef-flash-native-qualification-2026-10-02.md` for results.
